def process_route_optimization(geofence_str, house_coords, current_location, dump_coords, batch_size):
    import pandas as pd
    import networkx as nx
    import osmnx as ox
    from shapely.geometry import Point, Polygon
    from shapely.ops import nearest_points
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    fence_coords = [tuple(map(float, p.split(','))) for p in geofence_str.split(';')]
    polygon = Polygon([(lon, lat) for lat, lon in fence_coords])

    max_dist_m = 0
    for dlat, dlon in dump_coords:
        dump_point = Point(dlon, dlat)
        nearest_on_poly = nearest_points(polygon, dump_point)[0]
        dist_m = dump_point.distance(nearest_on_poly) * 111139
        max_dist_m = max(max_dist_m, dist_m)
    buffer_deg = (max_dist_m + 100) / 111139
    polygon_buffered = polygon.buffer(buffer_deg)

    G = ox.graph_from_polygon(polygon_buffered, network_type='drive')

    df = pd.DataFrame(house_coords)
    df['inside'] = df.apply(lambda r: polygon.covers(Point(r['lon'], r['lat'])), axis=1)
    df = df[df['inside']].reset_index(drop=True)
    df['HouseID'] = df.index

    house_points = {
        row.HouseID: (row.lat, row.lon, getattr(row, 'house_id', f"House-{row.HouseID}"))
        for row in df.itertuples()
    }
    house_queue = list(house_points.items())

    depot_coord = (current_location['lat'], current_location['lon']) if current_location else (polygon.centroid.y, polygon.centroid.x)
    current_coord = depot_coord

    batches = []
    batch_idx = 0

    while house_queue:
        current_batch = house_queue[:batch_size]
        house_queue = house_queue[batch_size:]

        batch_nodes = [ox.nearest_nodes(G, lon, lat) for _, (lat, lon, _) in current_batch]
        start_node = ox.nearest_nodes(G, current_coord[1], current_coord[0])
        nodes = [start_node] + batch_nodes

        size = len(nodes)
        matrix = [[0]*size for _ in range(size)]
        for i in range(size):
            dists = nx.single_source_dijkstra_path_length(G, nodes[i], weight='length')
            for j in range(size):
                matrix[i][j] = int(dists.get(nodes[j], float('inf')))

        mgr = pywrapcp.RoutingIndexManager(size, 1, 0)
        tsp = pywrapcp.RoutingModel(mgr)

        def dist_cb(i, j): return matrix[mgr.IndexToNode(i)][mgr.IndexToNode(j)]
        transit = tsp.RegisterTransitCallback(dist_cb)
        tsp.SetArcCostEvaluatorOfAllVehicles(transit)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.time_limit.seconds = 5

        sol = tsp.SolveWithParameters(params)

        batch_visit_sequence = []
        if batch_idx == 0:
            batch_visit_sequence.append(('Depot', current_coord, 'Depot'))

        if sol:
            idx = tsp.Start(0)
            while not tsp.IsEnd(idx):
                node = mgr.IndexToNode(idx)
                if node > 0:
                    hid, (lat, lon, house_id) = current_batch[node - 1]
                    batch_visit_sequence.append((hid, (lat, lon), house_id))
                    current_coord = (lat, lon)
                idx = sol.Value(tsp.NextVar(idx))

        # Dump yard selection
        dump_coord = None
        min_dist = float('inf')
        for dlat, dlon in dump_coords:
            try:
                dnode = ox.nearest_nodes(G, dlon, dlat)
                path = nx.shortest_path(G, nodes[-1], dnode, weight='length')
                dist = sum(G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:]))
                if dist < min_dist:
                    min_dist = dist
                    dump_coord = (dlat, dlon)
            except:
                continue

        if dump_coord:
            batch_visit_sequence.append((f"Dump-{batch_idx+1}", dump_coord, "Dump Yard"))
            current_coord = dump_coord  # IMPORTANT: Start next batch from here
        else:
            batch_visit_sequence.append((f"Dump-{batch_idx+1}", current_coord, "Dump Yard"))

        # Build route path and stats
        batch_route_path = []
        total_dist = 0
        for a, b in zip(batch_visit_sequence, batch_visit_sequence[1:]):
            coord_a, coord_b = a[1], b[1]
            try:
                na = ox.nearest_nodes(G, coord_a[1], coord_a[0])
                nb = ox.nearest_nodes(G, coord_b[1], coord_b[0])
                path = nx.shortest_path(G, na, nb, weight='length')
                seg = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
                dist = sum(G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:]))
                batch_route_path.extend(seg)
                total_dist += dist
            except:
                batch_route_path.extend([coord_a, coord_b])

        # Stops for frontend
        stops = [
            {
                "stop": idx,
                "label": label,
                "house_id": house_id,
                "lat": lat,
                "lon": lon
            }
            for idx, (label, (lat, lon), house_id) in enumerate(batch_visit_sequence)
        ]

        total_km = total_dist / 1000
        wait_time_min = 0 * sum(1 for s in batch_visit_sequence if "House" in str(s[2]))
        speed_profiles = []
        for spd in [10, 20, 30, 40]:
            drive_time = (total_km / spd) * 60
            total_time = drive_time + wait_time_min
            speed_profiles.append({
                "speed_kmph": spd,
                "distance_km": round(total_km, 2),
                "time_minutes": round(total_time, 2)
            })

        # pathway = []
        # for i, (label, _, hid) in enumerate(batch_visit_sequence):
        #     if i == 0 and label == "Depot":
        #         pathway.append("Start at Depot")
        #     elif "Dump" in str(label):
        #         pathway.append("Go to Dump Yard")
        #     else:
        #         pathway.append(f"Stop {i}: Visit {hid}")

        batches.append({
            "batch_index": batch_idx + 1,
            "stops": stops,
            "route_path": [{"lat": lat, "lon": lon} for lat, lon in batch_route_path],
            # "pathway": pathway,
            "speed_profiles": speed_profiles
        })

        batch_idx += 1

    return {
        "status": "success",
        "batches": batches
    }
