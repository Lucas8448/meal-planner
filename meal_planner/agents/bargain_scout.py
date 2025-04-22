"""Agent responsible for finding best prices for missing ingredients."""

from textwrap import dedent
from typing import Dict, Any, List

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from meal_planner.config.settings import settings
from meal_planner.tools.kassalapp import search_products, get_product_details
from meal_planner.models.state import MealPlannerState
from meal_planner.utils.json_helpers import parse_llm_json_output


class IngredientPricingAgent:
    """Agent that finds optimal pricing options for missing ingredients across different stores."""
    
    def __init__(self):
        """Initialize the Ingredient Pricing agent."""
        self.prompt_template = dedent("""\
            You are Agent 3: The Bargain Scout. Your input is a list of generic missing ingredient names identified by Agent 2.
            Your goal is to find **one standard, reasonably priced option** for each missing ingredient, looking across all available nearby stores.

            Input Missing Ingredients: {missing_ingredients_list}

            Follow these steps **for each ingredient** in the input list:
            1. Call `search_products` for the ingredient name. **Crucially, set `filter_by_price_drop` to `False`**.
            2. Analyze the results from `search_products`. Identify 2-4 promising candidate products that seem like standard, common forms of the ingredient (e.g., prefer 'Løk 1kg' or 'Løk pk' over 'Sprøstekt Løk'; prefer 'Olivenolje 500ml' over 'Olivenolje med Chili').
            3. If promising candidates were found, call `get_product_details` for **each** candidate product ID to fetch detailed information, specifically looking for `product_id`, `name`, `current_price`, `store`, and `unit_measure_name`.
            4. From the candidates with details, select the **single best option**. Prioritize:
               a) Candidates with standard packaging/units (like 'kg', 'liter', 'pk' for garlic).
               b) Among those, the one with the **lowest `current_price`**.
               c) If multiple similar options exist, pick one reasonably priced one (e.g., from a common store like Rema 1000, KIWI, Coop Extra if available).
            5. If no products are found for an ingredient in step 1, or if no suitable candidates with details are found in step 3, skip that ingredient.

            After iterating through all ingredients:
            6. Prepare a JSON list containing dictionaries for each ingredient where you found a suitable option based on step 4. Each dictionary should include:
               - "ingredient_name": The original missing ingredient name.
               - "product_id": The ID of the selected product.
               - "product_name": The name of the selected product.
               - "store": The store where the selected product was found.
               - "current_price": The current price of the selected item.
               - "unit": The unit measure name (e.g., 'kg', 'l', 'stk', or null if unavailable).
            7. Respond ONLY with this JSON list. Do not add any explanations. If no options were found for any ingredients, respond with an empty JSON list `[]`.
        """)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_template),
            ("user", "Find the cheapest options for these missing ingredients: {missing_ingredients_list}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.tools = [search_products, get_product_details]
        self.llm = ChatOpenAI(
            model=settings.llm_model, 
            temperature=settings.llm_temperature
        )
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=False, 
            max_iterations=40
        )
    
    def run(self, state: MealPlannerState) -> MealPlannerState:
        """Run the Ingredient Pricing agent to find cheapest options for missing ingredients.
        
        Args:
            state: The current state of the meal planning workflow.
            
        Returns:
            Updated state with optimal pricing information for ingredients.
        """
        print("--- Running Agent 3: Ingredient Pricing ---")
        missing_ingredients = state.get("missing_ingredients", [])
        
        if not missing_ingredients:
            print("DEBUG (Agent 3): No missing ingredients identified by Agent 2. Skipping.")
            updated_state = state.copy()
            updated_state["agent_outcome"] = {"status": "skipped", "reason": "No missing ingredients"}
            updated_state["cheapest_ingredients_info"] = []
            return updated_state
        
        # Format list for the prompt
        missing_ingredients_str = ", ".join(missing_ingredients)
        
        # Invoke the agent
        response = self.executor.invoke({"missing_ingredients_list": missing_ingredients_str})
        agent_output = response.get("output", "[]")
        
        print(f"DEBUG (Agent 3 Output): {agent_output}")
        
        # Parse and validate the output
        parsed_output, error = parse_llm_json_output(agent_output)
        
        if error:
            updated_state = state.copy()
            updated_state["cheapest_ingredients_info"] = []
            updated_state["agent_outcome"] = {"error": error, "raw_output": agent_output}
            return updated_state
        
        try:
            # Agent should output a list directly
            if not isinstance(parsed_output, list):
                raise ValueError("Parsed JSON is not a list as expected.")
            
            # Basic validation of list items
            validated_info = []
            required_keys = {"ingredient_name", "product_id", "product_name", "store", "current_price", "unit"}
            for item in parsed_output:
                if isinstance(item, dict) and required_keys.issubset(item.keys()):
                    validated_info.append(item)
                else:
                    print(f"WARN (Agent 3): Skipping invalid item in output: {item}")
            
            print(f"DEBUG (Agent 3 Parsed): Found optimal prices for {len(validated_info)} ingredients.")
            
            # Update and return state
            updated_state = state.copy()
            updated_state["cheapest_ingredients_info"] = validated_info
            updated_state["agent_outcome"] = validated_info
            return updated_state
            
        except Exception as e:
            print(f"ERROR (Agent 3): {str(e)}")
            updated_state = state.copy()
            updated_state["cheapest_ingredients_info"] = []
            updated_state["agent_outcome"] = {"error": f"Agent 3 Error: {str(e)}", "raw_output": agent_output}
            return updated_state 