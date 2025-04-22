"""Pydantic models for the API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from meal_planner.models.state import MealPlannerState, MealPlanItem, ShoppingListItem


class PlanRequest(BaseModel):
    """Input model for the meal planning API."""
    
    query: str = "Find common dinner ingredients in Norway with recent price drops to help plan meals."
    on_hand_ingredients: Optional[List[str]] = Field(
        default_factory=list, 
        description="Optional list of ingredient names user already has."
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Find good dinner deals in Norway for a family of four",
                "on_hand_ingredients": ["pasta", "rice", "onions", "garlic", "olive oil", "salt", "pepper"]
            }
        }


class PlanResponse(BaseModel):
    """Concise response model for the meal planning API."""
    
    chosen_store: str = Field(description="The store selected for the meal plan")
    meal_plan: List[MealPlanItem] = Field(description="A list of 7 dinner ideas for the week")
    missing_ingredients: List[str] = Field(description="Essential ingredients missing for the recipes")
    shopping_list: Dict[str, List[ShoppingListItem]] = Field(description="Shopping list organized by store")
    
    @classmethod
    def from_state(cls, state: MealPlannerState) -> "PlanResponse":
        """Create a concise response from the full state."""
        return cls(
            chosen_store=state["chosen_store"] or "",
            meal_plan=state["meal_plan"],
            missing_ingredients=state["missing_ingredients"],
            shopping_list=state["shopping_list"]
        ) 