"""Application configuration settings"""
import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GPT_MODEL = os.environ.get('GPT_MODEL', 'gpt-4o-mini')
    
    # Dataset configuration
    DATASET_PATH = 'data/diabetes.csv'
    MODEL_PATH = 'models/diabetes_model_logistic_regression.pkl'
    
    # Cache configuration
    CACHE_DIR = 'cache'
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

class TestConfig(Config):
    """Test configuration"""
    TESTING = True
    DEBUG = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestConfig,
    'default': DevelopmentConfig
}