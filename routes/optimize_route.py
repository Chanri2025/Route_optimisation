from flask import Blueprint, request, jsonify
import requests
from route_optimizer import process_route_optimization  # Use absolute import

optimize_route_bp = Blueprint('optimize_route', __name__)


@optimize_route_bp.route('/optimize_route', methods=['POST'])
def optimize_route():
    """
    API endpoint to optimize a route based on geofence and house coordinates.
    Returns a JSON response with the optimized route data.
    """
    try:
        # Get parameters from request
        data = request.get_json()

        # Required parameters
        geofence_str = data.get('geofence')  # Format: "lat,lon;lat,lon;..."
        house_coords_raw = data.get('houses')  # Format: [{"lat": 21.1445, "lon": 79.0585}, ...]

        # Process house coordinates to include house_id if not already present
        house_coords = []
        for idx, house in enumerate(house_coords_raw):
            house_data = {
                'lat': house.get('lat'),
                'lon': house.get('lon')
            }

            # Use provided house_id or House_Id if available, otherwise use index
            if 'house_id' in house:
                house_data['house_id'] = house['house_id']
            elif 'House_Id' in house:
                house_data['house_id'] = house['House_Id']
            else:
                house_data['house_id'] = f"House-{idx + 1}"

            house_coords.append(house_data)

        # Optional parameters
        nn_steps = data.get('nn_steps', 0)
        manual_center = data.get('center')  # Format: [lat, lon] or null

        # Get current location (if provided)
        current_location = data.get('current_location')  # Format: {"lat": 21.1445, "lon": 79.0585}

        # If current location not provided, try to get it from client IP
        if not current_location:
            try:
                # Get client IP address
                if request.headers.getlist("X-Forwarded-For"):
                    client_ip = request.headers.getlist("X-Forwarded-For")[0]
                else:
                    client_ip = request.remote_addr

                # Use IP geolocation service to get location
                # Note: In production, use a more reliable service with API key
                if client_ip != '127.0.0.1' and client_ip != 'localhost':
                    geo_response = requests.get(f'https://ipapi.co/{client_ip}/json/')
                    if geo_response.status_code == 200:
                        location_data = geo_response.json()
                        current_location = {
                            'lat': location_data.get('latitude'),
                            'lon': location_data.get('longitude')
                        }
            except Exception as e:
                print(f"Error getting location from IP: {e}")
                # Continue without current location if there's an error

        # Process the route optimization
        result = process_route_optimization(
            geofence_str=geofence_str,
            house_coords=house_coords,
            nn_steps=nn_steps,
            manual_center=manual_center,
            current_location=current_location
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500