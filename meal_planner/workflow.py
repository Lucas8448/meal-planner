"""LangGraph workflow for the meal planner application."""

from langgraph.graph import StateGraph, END

from meal_planner.models.state import MealPlannerState
from meal_planner.agents.agents_registry import (
    get_product_search_agent,
    get_meal_planning_agent, 
    get_ingredient_pricing_agent,
    get_shopping_list_agent
)


def create_workflow() -> StateGraph:
    """Create and configure the LangGraph workflow.
    
    Returns:
        Compiled StateGraph ready for execution.
    """
    # Define node functions that use the agents registry
    def run_product_search(state: MealPlannerState) -> MealPlannerState:
        return get_product_search_agent().run(state)
    
    def run_meal_planning(state: MealPlannerState) -> MealPlannerState:
        return get_meal_planning_agent().run(state)
    
    def run_ingredient_pricing(state: MealPlannerState) -> MealPlannerState:
        return get_ingredient_pricing_agent().run(state)
    
    def run_shopping_list(state: MealPlannerState) -> MealPlannerState:
        return get_shopping_list_agent().run(state)
    
    # Create the graph
    workflow = StateGraph(MealPlannerState)
    
    # Add nodes
    workflow.add_node("product_search", run_product_search)
    workflow.add_node("meal_planning", run_meal_planning)
    workflow.add_node("ingredient_pricing", run_ingredient_pricing)
    workflow.add_node("shopping_list_creator", run_shopping_list)
    
    # Define edges
    workflow.set_entry_point("product_search")
    workflow.add_edge("product_search", "meal_planning")
    workflow.add_edge("meal_planning", "ingredient_pricing")
    workflow.add_edge("ingredient_pricing", "shopping_list_creator")
    workflow.add_edge("shopping_list_creator", END)
    
    # Compile the graph
    return workflow.compile()


# Create a singleton instance of the compiled workflow
workflow_app = create_workflow() 