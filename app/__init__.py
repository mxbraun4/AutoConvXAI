"""Flask application initialization"""
from flask import Flask

def create_app():
    """Application factory pattern"""
    app = Flask(__name__, template_folder='../ui/templates', static_folder='../ui/static')
    
    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app