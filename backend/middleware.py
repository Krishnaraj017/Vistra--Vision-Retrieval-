from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def setup_middleware(app: FastAPI) -> None:
    """Configure middleware for the FastAPI application"""
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins - for development only!
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )