import pandas as pd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def process_route_optimization(geofence_str, house_coords, nn_steps=0, manual_center=None, current_location=None):
    if not geofence_str or not house_coords:
        return {
            "status": "error",
            "message": "Missing required fields: 'geofence' and/or 'houses'"
        }

    try:
        fence_coords = [tuple(map(float, p.split(','))) for p in geofence_str.split(';')]
    except Exception as e:
        return {
            "status": "error",
            "message": f"Invalid geofence format: {e}"
        }

    polygon = Polygon([(lon, lat) for lat, lon in fence_coords])

    # Process house DataFrame
    df_h = pd.DataFrame(house_coords)
    lat_col = 'lat' if 'lat' in df_h.columns else [c for c in df_h.columns if 'lat' in c.lower()][0]
    lon_col = 'lon' if 'lon' in df_h.columns else [c for c in df_h.columns if 'lon' in c.lower()][0]
    if 'house_id' not in df_h.columns:
        df_h['house_id'] = df_h.index
    df_h['inside'] = df_h.apply(lambda r: polygon.covers(Point(r[lon_col], r[lat_col])), axis=1)
    df_h = df_h[df_h['inside']].reset_index(drop=True)
    df_h['HouseID'] = df_h.index

    # Default buffer
    base_buffer_km = 0.1  # 100 meters

    # If a depot was provided, calculate how far it is from the polygon
    if current_location and 'lat' in current_location and 'lon' in current_location:
        depot_point = Point(current_location['lon'], current_location['lat'])
        nearest_on_polygon = nearest_points(depot_point, polygon)[1]
        distance_deg = depot_point.distance(nearest_on_polygon)
        buffer_deg = max(base_buffer_km / 111, distance_deg + 0.001)  # add 100m safety
        print(f"[INFO] Using buffer: {buffer_deg:.6f} degrees (~{buffer_deg * 111:.1f} meters)")
    else:
        buffer_deg = base_buffer_km / 111

    # Build the OSM graph using dynamic buffer
    G = ox.graph_from_polygon(polygon.buffer(buffer_deg), network_type='drive')


    # Determine depot
    if current_location and 'lat' in current_location and 'lon' in current_location:
        depot_coord = (current_location['lat'], current_location['lon'])
        current_point = Point(current_location['lon'], current_location['lat'])
        
    else:
        depot_pt = polygon.centroid
        depot_coord = (depot_pt.y, depot_pt.x)

    try:
        depot_node = ox.nearest_nodes(G, depot_coord[1], depot_coord[0])
    except:
        return {
            "status": "error",
            "message": "Depot location is unreachable on the road network."
        }

    # Snap houses and filter unreachable ones
    house_points = {}
    house_nodes = {}
    unreachable_houses = []

    for row in df_h.itertuples():
        lat, lon = getattr(row, lat_col), getattr(row, lon_col)
        try:
            node = ox.nearest_nodes(G, lon, lat)
            house_points[row.HouseID] = (lat, lon, row.house_id)
            house_nodes[row.HouseID] = node
        except:
            unreachable_houses.append({"house_id": row.house_id, "lat": lat, "lon": lon})
            print(f"[SKIP] House {row.house_id} at ({lat},{lon}) not reachable by road.")

    if not house_points:
        return {
            "status": "error",
            "message": "No reachable house points found in the road network."
        }

    # Nearest-neighbor seeding
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

    # TSP with OR-Tools
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
                matrix[i][j] = int(dlocs.get(sub_nodes[j], float('inf')))

        mgr = pywrapcp.RoutingIndexManager(size, 1, 0)
        tsp = pywrapcp.RoutingModel(mgr)

        def dist_cb(i, j): return matrix[mgr.IndexToNode(i)][mgr.IndexToNode(j)]
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

    # Build route path
    route_coords = [coord for _, coord, _ in visit_sequence]
    full_route_path = []
    total_distance_meters = 0

    for a, b in zip(route_coords, route_coords[1:]):
        try:
            n1 = ox.nearest_nodes(G, a[1], a[0])
            n2 = ox.nearest_nodes(G, b[1], b[0])
            path = nx.shortest_path(G, n1, n2, weight='length')
            segment = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
            full_route_path.extend(segment)
            segment_lengths = [G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:])]
            total_distance_meters += sum(segment_lengths)
        except:
            print(f"[WARNING] No road path from {a} to {b} — segment skipped.")

    full_route_coords = [{"lat": lat, "lon": lon} for lat, lon in full_route_path]

    # Speed profile
    total_distance_km = total_distance_meters / 1000
    num_stops = len(visit_sequence) - 2
    wait_time_minutes = 0 * num_stops
    speed_profiles = []
    for speed in [10, 20, 30, 40]:
        drive_time_min = (total_distance_km / speed) * 60
        total_time_min = drive_time_min + wait_time_minutes
        speed_profiles.append({
            "speed_kmph": speed,
            "distance_km": round(total_distance_km, 2),
            "time_minutes": round(total_time_min, 2)
        })

    # Stops + pathway
    stops_data = []
    pathway = []
    for idx, (label, (lat, lon), house_id) in enumerate(visit_sequence):
        if idx == 0:
            pathway.append("Start at Depot")
        elif idx == len(visit_sequence) - 1:
            pathway.append("Return to Depot")
        else:
            pathway.append(f"Stop {idx}: Visit {house_id}")

        stops_data.append({
            "Stop": idx,
            "Label": 'Depot' if label == 'Depot' else f'House {label}',
            "House_ID": house_id,
            "Latitude": lat,
            "Longitude": lon,
            "reachable": True  # these are all reachable by design
        })

    for h in unreachable_houses:
        stops_data.append({
            "Stop": None,
            "Label": "Unreachable",
            "House_ID": h["house_id"],
            "Latitude": h["lat"],
            "Longitude": h["lon"],
            "reachable": False
        })

    g_coords = [depot_coord] + [coord for _, coord, _ in visit_sequence[1:-1]] + [depot_coord]
    gmap_url = "https://www.google.com/maps/dir/" + "/".join(f"{lat},{lon}" for lat, lon in g_coords)

    return {
        "status": "success",
        "stops": stops_data,
        "depot": {"lat": depot_coord[0], "lon": depot_coord[1]},
        "google_maps_url": gmap_url,
        "pathway": pathway,
        "route_path": full_route_coords,
        "speed_profiles": speed_profiles
    }
