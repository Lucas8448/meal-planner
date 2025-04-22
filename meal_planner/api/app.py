"""FastAPI application configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from meal_planner.api.routes import router
from meal_planner.config.settings import settings, validate_required_settings


# Create the FastAPI app
app = FastAPI(
    title="Meal Planner Agent API",
    description="API to run the multi-agent meal planning process based on grocery deals.",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Validate settings at startup."""
    validate_required_settings() 