# services/__init__.py
"""
This file makes the 'services' directory a Python package.
"""
from .database_setup import create_db_tables, get_db_session, Base, engine, SessionLocal # Export DB components
from .data_service import DataService, get_benchmark_data_static
from .analysis_service import AnalysisService
from .portfolio_analysis import PortfolioAnalysisService
from .statistical_analysis_service import StatisticalAnalysisService
from .stochastic_model_service import StochasticModelService
from .ai_model_service import AIModelService
from .auth_service import AuthService

__all__ = [
    "Base", # For models to import
    "engine", # Potentially for direct use or advanced setup
    "SessionLocal", # For creating sessions if needed outside get_db_session
    "get_db_session",
    "create_db_tables",
    "DataService",
    "get_benchmark_data_static",
    "AnalysisService",
    "PortfolioAnalysisService",
    "StatisticalAnalysisService",
    "StochasticModelService",
    "AIModelService",
    "AuthService",
]
