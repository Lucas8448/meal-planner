"""Registry for agent instances to avoid recreating them for each request."""

from meal_planner.agents.deal_hunter import ProductSearchAgent
from meal_planner.agents.meal_strategist import MealPlanningAgent
from meal_planner.agents.bargain_scout import IngredientPricingAgent
from meal_planner.agents.list_consolidator import ShoppingListAgent


class AgentsRegistry:
    """A registry for managing agent instances."""
    
    _product_search: ProductSearchAgent = None
    _meal_planning: MealPlanningAgent = None
    _ingredient_pricing: IngredientPricingAgent = None
    _shopping_list: ShoppingListAgent = None
    
    @classmethod
    def get_product_search_agent(cls) -> ProductSearchAgent:
        """Get or create the Product Search agent instance."""
        if cls._product_search is None:
            cls._product_search = ProductSearchAgent()
        return cls._product_search
    
    @classmethod
    def get_meal_planning_agent(cls) -> MealPlanningAgent:
        """Get or create the Meal Planning agent instance."""
        if cls._meal_planning is None:
            cls._meal_planning = MealPlanningAgent()
        return cls._meal_planning
    
    @classmethod
    def get_ingredient_pricing_agent(cls) -> IngredientPricingAgent:
        """Get or create the Ingredient Pricing agent instance."""
        if cls._ingredient_pricing is None:
            cls._ingredient_pricing = IngredientPricingAgent()
        return cls._ingredient_pricing
    
    @classmethod
    def get_shopping_list_agent(cls) -> ShoppingListAgent:
        """Get or create the Shopping List agent instance."""
        if cls._shopping_list is None:
            cls._shopping_list = ShoppingListAgent()
        return cls._shopping_list


# Convenience functions for direct access to agents
def get_product_search_agent() -> ProductSearchAgent:
    """Get the Product Search agent instance."""
    return AgentsRegistry.get_product_search_agent()


def get_meal_planning_agent() -> MealPlanningAgent:
    """Get the Meal Planning agent instance."""
    return AgentsRegistry.get_meal_planning_agent()


def get_ingredient_pricing_agent() -> IngredientPricingAgent:
    """Get the Ingredient Pricing agent instance."""
    return AgentsRegistry.get_ingredient_pricing_agent()


def get_shopping_list_agent() -> ShoppingListAgent:
    """Get the Shopping List agent instance."""
    return AgentsRegistry.get_shopping_list_agent() 