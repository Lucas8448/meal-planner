"""Entry point for the Meal Planner application."""

import uvicorn
from meal_planner.config.settings import settings
from meal_planner.api.app import app


if __name__ == "__main__":
    print("## Starting Meal Planner API Server...")
    uvicorn.run(
        "meal_planner.api.app:app", 
        host=settings.api_host, 
        port=settings.api_port, 
        reload=settings.api_reload
    )