from flask import Blueprint, request, Response
import requests
from route_optimizer import process_route_optimization  # Use absolute import

map_html_bp = Blueprint('map_html', __name__)


@map_html_bp.route('/get_map_html', methods=['POST'])
def get_map_html():
    """
    API endpoint to get the HTML map for a route based on geofence and house coordinates.
    Returns the HTML directly for rendering in a browser.
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
        current_location = data.get('current_location')  # Format: {"lat": 21.1445, "lon": 79.0585}

        # Process the route optimization
        result = process_route_optimization(
            geofence_str=geofence_str,
            house_coords=house_coords,
            nn_steps=nn_steps,
            manual_center=manual_center,
            current_location=current_location
        )

        # Return only the HTML map
        return Response(result['map_html'], mimetype='text/html')

    except Exception as e:
        return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>", 500