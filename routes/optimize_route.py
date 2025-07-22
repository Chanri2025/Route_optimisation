from flask import Blueprint, request, jsonify
import requests
from route_optimizer import process_route_optimization  # assumes updated signature

optimize_route_bp = Blueprint('optimize_route', __name__)


@optimize_route_bp.route('/optimize_route', methods=['POST'])
def optimize_route():
    data = request.get_json()

    # geofence & houses (unchanged)
    geofence_str = data.get('geofence')
    houses_raw = data.get('houses')

    # new fields
    start_loc = data.get('start_location')  # {"lat":…, "lon":…}
    dump_loc = data.get('dump_location')  # {"lat":…, "lon":…}
    batch_sz = data.get('batch_size', len(houses_raw))

    # basic validation
    if not (geofence_str and houses_raw and start_loc and dump_loc):
        return jsonify({
            "status": "error",
            "message": "Required: geofence, houses, start_location, dump_location"
        }), 400

    try:
        result = process_route_optimization(
            geofence_str=geofence_str,
            house_coords=houses_raw,
            nn_steps=data.get('nn_steps', 0),
            manual_center=data.get('center'),
            start_location=start_loc,
            dump_location=dump_loc,
            batch_size=batch_sz
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
