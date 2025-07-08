import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import folium
from shapely.geometry import Point, Polygon
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import requests
import io
import os
from datetime import datetime


def process_route_optimization(geofence_str, house_coords, nn_steps=0, manual_center=None, current_location=None):
    """
    Process route optimization based on geofence and house coordinates.

    Args:
        geofence_str (str): Semicolon-separated lat,lon coordinates defining the geofence
        house_coords (list): List of dictionaries with lat, lon, and house_id keys for houses
        nn_steps (int): Number of nearest neighbor steps to perform
        manual_center (list): Optional [lat, lon] for map centering
        current_location (dict): Optional {"lat": float, "lon": float} for starting point

    Returns:
        dict: Result containing map HTML, route sequence, and other data
    """
    # ─── 1) Parse geofence polygon ───────────────────────────────────────────
    fence_coords = [tuple(map(float, p.split(','))) for p in geofence_str.split(';')]
    polygon = Polygon([(lon, lat) for lat, lon in fence_coords])  # shapely expects (lon, lat)

    # ─── 2) Process house stops ───────────────────────────────────────────────
    # Create a DataFrame from the house coordinates
    df_h = pd.DataFrame(house_coords)

    # Ensure column names match expected format
    if 'lat' in df_h.columns and 'lon' in df_h.columns:
        lat_col = 'lat'
        lon_col = 'lon'
    else:
        # Try to find columns with 'lat' and 'lon' in their names
        lat_col = [c for c in df_h.columns if 'lat' in c.lower()][0]
        lon_col = [c for c in df_h.columns if 'lon' in c.lower()][0]

    # Check if house_id is provided, otherwise use index
    if 'house_id' in df_h.columns:
        id_col = 'house_id'
    else:
        df_h['house_id'] = df_h.index
        id_col = 'house_id'

    # Filter houses inside the geofence
    df_h['inside'] = df_h.apply(
        lambda r: polygon.covers(Point(r[lon_col], r[lat_col])),
        axis=1
    )
    df_h = df_h[df_h['inside']].reset_index(drop=True)
    df_h['HouseID'] = df_h.index

    # Create a dictionary with house points and their IDs
    house_points = {
        row.HouseID: (getattr(row, lat_col), getattr(row, lon_col), getattr(row, id_col))
        for row in df_h.itertuples()
    }

    # ─── 3) Download road network ───────────────────────────────────────────────
    print('Downloading OSM road network...')
    G = ox.graph_from_polygon(polygon, network_type='drive')

    # ─── 4) Determine depot location (starting point) ────────────────────────────
    if current_location and 'lat' in current_location and 'lon' in current_location:
        # Use provided current location as depot
        depot_coord = (current_location['lat'], current_location['lon'])
        # Check if current location is inside or near the geofence
        current_point = Point(current_location['lon'], current_location['lat'])
        if not polygon.covers(current_point):
            # If outside, find nearest point on polygon boundary
            nearest_point = polygon.exterior.interpolate(polygon.exterior.project(current_point))
            depot_coord = (nearest_point.y, nearest_point.x)
    else:
        # Fallback to polygon centroid if no current location
        depot_pt = polygon.centroid
        depot_coord = (depot_pt.y, depot_pt.x)

    # Snap depot to network
    depot_node = ox.nearest_nodes(G, depot_coord[1], depot_coord[0])

    # Snap houses to network nodes
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
    if len(remaining) + 1 <= MAX_LOCS:
        rem_limit = remaining
    else:
        rem_limit = remaining[:MAX_LOCS - 1]
        print(f"Warning: TSP on {len(rem_limit)} stops; {len(remaining) - len(rem_limit)} appended.")

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

    # ─── Add return to depot ────────────────────────────────────────────────────
    visit_sequence.append(('Depot', depot_coord, 'Ending Point'))

    # ─── 7) Draw map with two view options ─────────────────────────────────────
    center = manual_center if manual_center else depot_coord
    m = folium.Map(location=center, zoom_start=15, tiles=None)
    folium.TileLayer('OpenStreetMap', name='Map View', control=True).add_to(m)
    folium.TileLayer('Esri.WorldImagery', name='Satellite View', control=True).add_to(m)
    folium.Polygon(locations=fence_coords, color='blue', weight=2,
                   fill=True, fill_opacity=0.1).add_to(m)

    # Add special marker for depot/starting point
    folium.Marker(
        location=depot_coord,
        popup="Starting Point",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)

    # Add house markers with their IDs
    for hid, (lat, lon, house_id) in house_points.items():
        folium.CircleMarker(
            location=(lat, lon),
            radius=4,
            color='red',
            fill=True,
            fill_opacity=0.7,
            popup=f"House ID: {house_id}"
        ).add_to(m)

    # Draw the route
    route_coords = [coord for _, coord, _ in visit_sequence]
    for a, b in zip(route_coords, route_coords[1:]):
        n1 = ox.nearest_nodes(G, a[1], a[0])
        n2 = ox.nearest_nodes(G, b[1], b[0])
        try:
            path = nx.shortest_path(G, n1, n2, weight='length')
            seg = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
            folium.PolyLine(locations=seg, color='black', weight=3).add_to(m)
        except nx.NetworkXNoPath:
            # If no path found, draw a direct line
            folium.PolyLine(locations=[a, b], color='red', weight=3, dash_array='5').add_to(m)

    # Add numbered markers for the visit sequence
    for idx, (label, (lat, lon), house_id) in enumerate(visit_sequence):
        if idx > 0 and idx < len(visit_sequence) - 1:  # Skip depot markers (first and last)
            folium.Marker(
                location=(lat, lon),
                popup=f"Stop {idx}: {house_id}",
                icon=folium.DivIcon(
                    html=f"<div style='font-size:10px;color:white;"
                         f"background:red;border-radius:50%;width:24px;height:24px;"
                         f"text-align:center;line-height:24px;'>{idx}</div>"
                )
            ).add_to(m)

    folium.LayerControl().add_to(m)

    # Save map to a string
    map_html = m._repr_html_()

    # ─── 8) Create route sequence data ─────────────────────────────────────────────
    seq_df = pd.DataFrame([
        {"Stop": idx,
         "Label": 'Depot' if label == 'Depot' else f'House {label}',
         "House_ID": house_id,
         "Latitude": lat,
         "Longitude": lon}
        for idx, (label, (lat, lon), house_id) in enumerate(visit_sequence)
    ])

    # Convert to CSV string
    csv_string = seq_df.to_csv(index=False)

    # ─── 9) Generate Google Maps link ───────────────────────────────────────────
    g_coords = [depot_coord] + [coord for _, coord, _ in visit_sequence[1:-1]] + [depot_coord]
    gmap_url = "https://www.google.com/maps/dir/" + "/".join(f"{lat},{lon}" for lat, lon in g_coords)

    # ─── 10) Create pathway with House IDs ─────────────────────────────────────
    pathway = []
    for idx, (label, _, house_id) in enumerate(visit_sequence):
        if idx == 0:
            pathway.append(f"Start at Depot")
        elif idx == len(visit_sequence) - 1:
            pathway.append(f"Return to Depot")
        else:
            pathway.append(f"Stop {idx}: Visit {house_id}")

    # Prepare response data
    response = {
        "status": "success",
        "stops": seq_df.to_dict(orient='records'),
        "depot": {"lat": depot_coord[0], "lon": depot_coord[1]},
        "google_maps_url": gmap_url,
        "map_html": map_html,
        "route_sequence_csv": csv_string,
        "pathway": pathway
    }

    return response