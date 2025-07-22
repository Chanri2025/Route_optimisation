# route_optimizer.py

import math
import pandas as pd
from shapely.geometry import Point, Polygon
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


def process_route_optimization(
        geofence_str: str,
        house_coords: list,
        start_location: dict,
        dump_location: dict,
        batch_size: int = None,
        nn_steps: int = 0,
        manual_center=None
):
    """
    Splits houses into batches for dump trips and returns for each trip:
      - stops (start/dump + houses),
      - route_path (lat/lon sequence),
      - speed_profiles (at speeds 10,20,30,40 km/h)
    """
    # 1. Parse geofence into polygon
    try:
        pts = [tuple(map(float, p.split(","))) for p in geofence_str.split(";")]
        polygon = Polygon([(lon, lat) for lat, lon in pts])
    except Exception as e:
        return {"status": "error", "message": f"Invalid geofence: {e}"}

    # 2. Filter houses inside the geofence
    df = pd.DataFrame(house_coords)
    if not {"lat", "lon"}.issubset(df.columns):
        return {"status": "error", "message": "houses must include 'lat' & 'lon'"}
    if 'house_id' not in df.columns:
        df['house_id'] = df.index.astype(str)
    df['inside'] = df.apply(lambda r: polygon.covers(Point(r['lon'], r['lat'])), axis=1)
    df = df[df['inside']].reset_index(drop=True)
    if df.empty:
        return {"status": "error", "message": "No houses inside geofence"}

    # 3. Prepare coordinates: start, houses..., dump
    start = (start_location['lat'], start_location['lon'])
    dump = (dump_location['lat'], dump_location['lon'])
    houses = df[['house_id', 'lat', 'lon']].to_dict(orient='records')
    id_to_idx = {h['house_id']: i for i, h in enumerate(houses)}
    coords = [start] + [(h['lat'], h['lon']) for h in houses] + [dump]
    N = len(coords)

    # 4. Compute accurate Euclidean distance matrix (meters)
    #    Convert degree differences to meters: 1° lat ≈ 111000 m, 1° lon ≈ 111000 m at equator
    dist = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                dy_deg = coords[i][0] - coords[j][0]
                dx_deg = coords[i][1] - coords[j][1]
                dy_m = dy_deg * 111000
                dx_m = dx_deg * 111000 * math.cos(math.radians((coords[i][0] + coords[j][0]) / 2))
                dist[i][j] = math.hypot(dy_m, dx_m)

    # 5. Create batches of house_ids
    all_ids = [h['house_id'] for h in houses]
    if not batch_size or batch_size < 1:
        batch_size = len(all_ids)
    batches = [all_ids[i:i + batch_size] for i in range(0, len(all_ids), batch_size)]

    results = []
    for b_idx, batch in enumerate(batches):
        # subproblem node indices: [0] start, [1..] houses, [M-1] dump
        subs = [0] + [1 + id_to_idx[hid] for hid in batch] + [N - 1]
        M = len(subs)
        sub_dist = [[dist[subs[i]][subs[j]] for j in range(M)] for i in range(M)]

        # 6. OR-Tools TSP with distinct start/end
        mgr = pywrapcp.RoutingIndexManager(M, 1, [0], [M - 1])
        routing = pywrapcp.RoutingModel(mgr)
        transit_callback_idx = routing.RegisterTransitCallback(
            lambda i, j: int(sub_dist[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
        )
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        params.time_limit.seconds = 5
        sol = routing.SolveWithParameters(params)

        # 7. Extract ordered house_ids
        order = []
        if sol:
            index = routing.Start(0)
            while not routing.IsEnd(index):
                node = mgr.IndexToNode(index)
                if 1 <= node < M - 1:
                    order.append(batch[node - 1])
                index = sol.Value(routing.NextVar(index))
        else:
            order = batch[:]

        # 8. Build stops list
        stops = []
        # start or dump
        stops.append({
            'stop': 0,
            'label': 'Start' if b_idx == 0 else 'Dump Yard',
            'house_id': None,
            'lat': coords[0][0],
            'lon': coords[0][1]
        })
        # houses
        for s, hid in enumerate(order, start=1):
            idx = subs[s]
            stops.append({
                'stop': s,
                'label': f'House {hid}',
                'house_id': hid,
                'lat': coords[idx][0],
                'lon': coords[idx][1]
            })
        # final dump
        stops.append({
            'stop': len(stops),
            'label': 'Dump Yard',
            'house_id': None,
            'lat': dump[0],
            'lon': dump[1]
        })

        # 9. route_path: direct sequence of subs
        route_path = [{'lat': coords[i][0], 'lon': coords[i][1]} for i in subs]

        # 10. compute total distance for speed profiles
        total_m = 0.0
        prev_idx = 0
        for hid in order:
            curr_idx = subs.index(1 + id_to_idx[hid])
            total_m += sub_dist[prev_idx][curr_idx]
            prev_idx = curr_idx
        total_m += sub_dist[prev_idx][M - 1]
        total_km = total_m / 1000.0

        # 11. speed_profiles at 10,20,30,40
        speed_profiles = []
        for sp in (10, 20, 30, 40):
            t_min = (total_km / sp) * 60
            speed_profiles.append({
                'speed_kmph': sp,
                'distance_km': round(total_km, 2),
                'time_minutes': round(t_min, 2)
            })

        results.append({
            'batch_index': b_idx,
            'stops': stops,
            'route_path': route_path,
            'speed_profiles': speed_profiles
        })

    return {'status': 'success', 'batches': results}
