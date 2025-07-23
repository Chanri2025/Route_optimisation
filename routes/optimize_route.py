from flask import Blueprint, request, jsonify
from route_optimizer import process_route_optimization

optimize_route_bp = Blueprint('optimize_route', __name__)


@optimize_route_bp.route('/optimize_route', methods=['POST'])
def optimize_route():
    data = request.get_json()

    # Required input fields
    geofence_str = data.get('geofence')
    houses_raw = data.get('houses')
    start_loc = data.get('start_location')  # e.g., {"lat": ..., "lon": ...}
    dump_loc = data.get('dump_location')    # e.g., {"lat": ..., "lon": ...}
    batch_sz = data.get('batch_size', len(houses_raw))

    # Basic validation
    if not (geofence_str and houses_raw and start_loc and dump_loc):
        return jsonify({
            "status": "error",
            "message": "Missing required fields: geofence, houses, start_location, dump_location"
        }), 400

    # Parse dump location into list of [lat, lon]
    dump_coords = [[dump_loc['lat'], dump_loc['lon']]]

    # Standardize house objects
    house_coords = []
    for idx, house in enumerate(houses_raw):
        house_data = {
            'lat': house.get('lat'),
            'lon': house.get('lon'),
            'house_id': house.get('house_id') or house.get('House_Id') or f"House-{idx + 1}"
        }
        house_coords.append(house_data)

    try:
        result = process_route_optimization(
            geofence_str=geofence_str,
            house_coords=house_coords,
            current_location=start_loc,
            dump_coords=dump_coords,
            batch_size=batch_sz
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
