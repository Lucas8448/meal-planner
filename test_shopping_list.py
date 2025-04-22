"""Test script for the ShoppingListAgent."""

from meal_planner.agents.list_consolidator import ShoppingListAgent
from meal_planner.models.state import MealPlannerState

def test_shopping_list_agent():
    """Test the ShoppingListAgent directly."""
    # Create a minimal test state
    test_state = MealPlannerState(
        initial_query="Test query",
        on_hand_ingredients=["onions", "garlic", "olive oil"],
        search_terms=[],
        found_deals=[],
        consolidated_deals=[],
        chosen_store="REMA 1000",
        meal_plan=[
            {
                "meal_name": "Test Meal 1",
                "deals_used": [
                    {"id": 123, "name": "Test Product", "current_price": 49.9, "currency": "NOK", "store": "REMA 1000"}
                ],
                "on_hand_used": ["onions", "garlic"]
            }
        ],
        missing_ingredients=["butter"],
        cheapest_ingredients_info=[
            {
                "ingredient_name": "butter",
                "product_id": 456,
                "product_name": "Test Butter",
                "store": "KIWI",
                "current_price": 39.9,
                "unit": "stk"
            }
        ],
        shopping_list={},
        agent_outcome=None
    )
    
    # Create and run the agent
    agent = ShoppingListAgent()
    result_state = agent.run(test_state)
    
    # Print the results
    print("\nTest results:")
    print(f"Shopping list: {result_state.get('shopping_list')}")
    print(f"Agent outcome: {result_state.get('agent_outcome')}")

if __name__ == "__main__":
    print("Testing ShoppingListAgent...")
    test_shopping_list_agent()
    print("Test complete") 