import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, Polygon
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from datetime import datetime

def process_route_optimization(geofence_str, house_coords, nn_steps=0, manual_center=None, current_location=None):
    # ─── 1) Parse geofence polygon ───────────────────────────────────────────
    fence_coords = [tuple(map(float, p.split(','))) for p in geofence_str.split(';')]
    polygon = Polygon([(lon, lat) for lat, lon in fence_coords])

    # ─── 2) Process house stops ───────────────────────────────────────────────
    df_h = pd.DataFrame(house_coords)

    lat_col = 'lat' if 'lat' in df_h.columns else [c for c in df_h.columns if 'lat' in c.lower()][0]
    lon_col = 'lon' if 'lon' in df_h.columns else [c for c in df_h.columns if 'lon' in c.lower()][0]

    if 'house_id' in df_h.columns:
        id_col = 'house_id'
    else:
        df_h['house_id'] = df_h.index
        id_col = 'house_id'

    df_h['inside'] = df_h.apply(
        lambda r: polygon.covers(Point(r[lon_col], r[lat_col])),
        axis=1
    )
    df_h = df_h[df_h['inside']].reset_index(drop=True)
    df_h['HouseID'] = df_h.index

    house_points = {
        row.HouseID: (getattr(row, lat_col), getattr(row, lon_col), getattr(row, id_col))
        for row in df_h.itertuples()
    }

    # ─── 3) Download road network ───────────────────────────────────────────────
    print('Downloading OSM road network...')
    G = ox.graph_from_polygon(polygon, network_type='drive')

    # ─── 4) Determine depot location ────────────────────────────────────────────
    if current_location and 'lat' in current_location and 'lon' in current_location:
        depot_coord = (current_location['lat'], current_location['lon'])
        current_point = Point(current_location['lon'], current_location['lat'])
        if not polygon.covers(current_point):
            nearest_point = polygon.exterior.interpolate(polygon.exterior.project(current_point))
            depot_coord = (nearest_point.y, nearest_point.x)
    else:
        depot_pt = polygon.centroid
        depot_coord = (depot_pt.y, depot_pt.x)

    depot_node = ox.nearest_nodes(G, depot_coord[1], depot_coord[0])

    house_nodes = {
        hid: ox.nearest_nodes(G, lon, lat)
        for hid, (lat, lon, _) in house_points.items()
    }

    # ─── 5) Nearest-Neighbor seed ───────────────────────────────────────────────
    visit_sequence = [('Depot', depot_coord, 'Starting Point')]
    visited = set()
    current_node = depot_node

    for _ in range(nn_steps):
        unvisited = [hid for hid in house_points if hid not in visited]
        if not unvisited:
            break
        dists = nx.single_source_dijkstra_path_length(G, current_node, weight='length')
        next_hid = min(unvisited, key=lambda hid: dists.get(house_nodes[hid], float('inf')))
        visited.add(next_hid)
        current_node = house_nodes[next_hid]
        visit_sequence.append((next_hid, house_points[next_hid][:2], house_points[next_hid][2]))

    # ─── 6) OR-Tools TSP on remaining ────────────────────────────────────────────
    remaining = [hid for hid in house_points if hid not in visited]
    MAX_LOCS = 59
    rem_limit = remaining[:MAX_LOCS - 1] if len(remaining) + 1 > MAX_LOCS else remaining

    if rem_limit:
        sub_nodes = [current_node] + [house_nodes[hid] for hid in rem_limit]
        size = len(sub_nodes)
        matrix = [[0] * size for _ in range(size)]
        for i in range(size):
            dlocs = nx.single_source_dijkstra_path_length(G, sub_nodes[i], weight='length')
            for j in range(size):
                matrix[i][j] = int(dlocs.get(sub_nodes[j], 0))

        mgr = pywrapcp.RoutingIndexManager(size, 1, 0)
        tsp = pywrapcp.RoutingModel(mgr)

        def dist_cb(i, j):
            return matrix[mgr.IndexToNode(i)][mgr.IndexToNode(j)]

        transit = tsp.RegisterTransitCallback(dist_cb)
        tsp.SetArcCostEvaluatorOfAllVehicles(transit)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.time_limit.seconds = 10

        sol = tsp.SolveWithParameters(params)
        if sol:
            idx = tsp.Start(0)
            idx = sol.Value(tsp.NextVar(idx))
            while not tsp.IsEnd(idx):
                node = mgr.IndexToNode(idx)
                if node > 0:
                    hid = rem_limit[node - 1]
                    visit_sequence.append((hid, house_points[hid][:2], house_points[hid][2]))
                idx = sol.Value(tsp.NextVar(idx))

    for hid in remaining[len(rem_limit):]:
        visit_sequence.append((hid, house_points[hid][:2], house_points[hid][2]))

    visit_sequence.append(('Depot', depot_coord, 'Ending Point'))

    # ─── 7) Construct full road-following path ───────────────────────────────────
    full_route_path = []
    route_coords = [coord for _, coord, _ in visit_sequence]

    for a, b in zip(route_coords, route_coords[1:]):
        n1 = ox.nearest_nodes(G, a[1], a[0])
        n2 = ox.nearest_nodes(G, b[1], b[0])
        try:
            path = nx.shortest_path(G, n1, n2, weight='length')
            segment = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
            full_route_path.extend(segment)
        except nx.NetworkXNoPath:
            full_route_path.append((a[0], a[1]))
            full_route_path.append((b[0], b[1]))

    full_route_coords = [
        {"lat": lat, "lon": lon} for lat, lon in full_route_path
    ]

    # ─── 8) Pathway description ──────────────────────────────────────────────────
    pathway = []
    for idx, (label, _, house_id) in enumerate(visit_sequence):
        if idx == 0:
            pathway.append(f"Start at Depot")
        elif idx == len(visit_sequence) - 1:
            pathway.append(f"Return to Depot")
        else:
            pathway.append(f"Stop {idx}: Visit {house_id}")

    # ─── 9) Stops summary ────────────────────────────────────────────────────────
    stops_data = [
        {"Stop": idx,
         "Label": 'Depot' if label == 'Depot' else f'House {label}',
         "House_ID": house_id,
         "Latitude": lat,
         "Longitude": lon}
        for idx, (label, (lat, lon), house_id) in enumerate(visit_sequence)
    ]

    # ─── 10) Google Maps link ────────────────────────────────────────────────────
    g_coords = [depot_coord] + [coord for _, coord, _ in visit_sequence[1:-1]] + [depot_coord]
    gmap_url = "https://www.google.com/maps/dir/" + "/".join(f"{lat},{lon}" for lat, lon in g_coords)

    # ─── Final Response ──────────────────────────────────────────────────────────
    response = {
        "status": "success",
        "stops": stops_data,
        "depot": {"lat": depot_coord[0], "lon": depot_coord[1]},
        "google_maps_url": gmap_url,
        "pathway": pathway,
        "route_path": full_route_coords
    }

    return response
