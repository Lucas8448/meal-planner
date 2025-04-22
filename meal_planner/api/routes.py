"""API route definitions for the meal planner application."""

import asyncio
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from meal_planner.api.models import PlanRequest, PlanResponse
from meal_planner.models.state import MealPlannerState
from meal_planner.workflow import workflow_app


# Create router
router = APIRouter(prefix="/api", tags=["meal-planner"])


@router.post("/plan-meals", response_model=PlanResponse)
async def run_meal_plan(request: PlanRequest) -> PlanResponse:
    """Run the meal planning workflow based on the user's request.
    
    Args:
        request: The meal planning request with query and on-hand ingredients.
        
    Returns:
        The complete meal plan with shopping list in a concise format.
        
    Raises:
        HTTPException: If the workflow execution fails.
    """
    print(f"--- API Request Received: /plan-meals ---")
    print(f"Query: {request.query}")
    print(f"On Hand: {request.on_hand_ingredients}")
    
    # Initialize state
    initial_state: MealPlannerState = {
        "initial_query": request.query,
        "on_hand_ingredients": request.on_hand_ingredients or [],
        "search_terms": [],
        "found_deals": [],
        "consolidated_deals": [],
        "chosen_store": None,
        "meal_plan": [],
        "missing_ingredients": [],
        "cheapest_ingredients_info": [],
        "shopping_list": {},
        "agent_outcome": None,
    }
    
    try:
        # Run the workflow in a separate thread to avoid blocking
        # the FastAPI event loop with synchronous code
        final_state = await asyncio.to_thread(workflow_app.invoke, initial_state)
        print("--- API Request Completed Successfully ---")
        
        # Convert to concise response
        return PlanResponse.from_state(final_state)
    except Exception as e:
        print(f"ERROR during graph invocation: {e}")
        # Return a proper HTTP error
        raise HTTPException(
            status_code=500, 
            detail=f"Meal planning workflow failed: {str(e)}"
        ) 