from flask import Flask
from flask_cors import CORS
from routes.optimize_route import optimize_route_bp
from routes.waste_classification import waste_classification_bp


def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

    # Register blueprints
    app.register_blueprint(optimize_route_bp)
    app.register_blueprint(waste_classification_bp)

    # Add root route
    @app.route('/')
    def index():
        return {"status": "running", "message": "Flask app is up and running!"}

    return app


# This line makes 'app' available for waitress-serve
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
