"""State definitions for the meal planning workflow."""

from typing import Dict, List, Optional, Union, TypedDict, Any


class DealInfo(TypedDict):
    """Detailed information about a product deal."""
    id: int
    name: str
    current_price: float
    previous_price: Optional[float]
    price_drop_percentage: Optional[float]
    currency: str
    store: str
    image_url: Optional[str]


class MealPlanItem(TypedDict):
    """Structure for a single meal in the plan."""
    meal_name: str
    deals_used: List[DealInfo]
    on_hand_used: List[str]
    notes: str


class ShoppingListItem(TypedDict):
    """Structure for an item in the final shopping list."""
    name: str
    price: float
    currency: str
    notes: str
    image_url: str


class MealPlannerState(TypedDict):
    """
    Represents the state of our meal planning graph.
    """
    initial_query: str
    on_hand_ingredients: List[str]
    search_terms: List[str]
    found_deals: List[DealInfo]
    consolidated_deals: List[DealInfo]
    chosen_store: Optional[str]
    meal_plan: List[MealPlanItem]
    missing_ingredients: List[str]
    cheapest_ingredients_info: List[Dict[str, Any]]
    shopping_list: Dict[str, List[ShoppingListItem]]
    agent_outcome: Dict[str, Any] 