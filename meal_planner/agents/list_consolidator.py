"""Agent responsible for creating the final shopping list by store."""

import json
from textwrap import dedent
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from meal_planner.config.settings import settings
from meal_planner.models.state import MealPlannerState
from meal_planner.utils.json_helpers import parse_llm_json_output


class ShoppingListAgent:
    """Agent that creates an organized shopping list based on meal plan and missing ingredients."""
    
    def __init__(self):
        """Initialize the Shopping List agent."""
        self.prompt_template = dedent("""\
            You are Agent 4: The List Consolidator. Your goal is to create a final shopping list for ONLY the chosen store.

            You have the following information:
            - The chosen store for the main deals: {chosen_store}
            - The meal plan (containing the specific deal items used for each meal):
            ```json
            {meal_plan_json}
            ```
            - The list of reasonably priced options found for missing ingredients (potentially from various stores):
            ```json
            {cheapest_missing_json}
            ```

            Follow these steps:
            1. Extract all the **deal items** directly from the `deals_used` lists within the `meal_plan_json`. ONLY include items where the store name matches exactly the `chosen_store` value.
            2. IGNORE all items from the `cheapest_missing_json` list that are not from the exact same store as `chosen_store`.
            3. Combine the remaining items from step 1 and step 2 into a single shopping list for ONLY the chosen store.
            4. Your output should be a JSON object with a SINGLE key (the chosen store) with a list of items to buy at that store only.
            5. Each item in the store list MUST be a dictionary containing EXACTLY the following fields (all are required):
               - `name`: Product name.
               - `price`: The `current_price` of the item.
               - `currency`: Currency code (e.g., "NOK").
               - `notes`: Add "Deal item" for items from step 1. For items from step 2, add "Staple item for [ingredient_name]".
               - `image_url`: ALWAYS include this field. Use the original image_url if available, otherwise set to null.

            6. Respond ONLY with the final JSON shopping list object with the SINGLE chosen store as the only key. Do not add explanations. If no items are found for the chosen store, respond with `{{}}`.

            IMPORTANT: 
            - Make sure EVERY item in the shopping list includes ALL required fields.
            - ONLY include items from the chosen store - do not include items from other stores.
            - The output should have exactly ONE store key matching the chosen_store value.

            Example output format:
            ```json
            {{
              "SPAR": [
                {{
                  "name": "Product Name",
                  "price": 49.9,
                  "currency": "NOK",
                  "notes": "Deal item",
                  "image_url": "https://example.com/image.jpg"
                }},
                ...
              ]
            }}
            ```
        """)
        
        self.llm = ChatOpenAI(
            model=settings.llm_model, 
            temperature=settings.llm_temperature
        )
        self.chain = ChatPromptTemplate.from_template(self.prompt_template) | self.llm
    
    def run(self, state: MealPlannerState) -> MealPlannerState:
        """Run the Shopping List agent to create the final organized shopping list.
        
        Args:
            state: The current state of the meal planning workflow.
            
        Returns:
            Updated state with consolidated shopping list organized by store.
        """
        print("--- Running Agent 4: Shopping List ---")
        
        meal_plan = state.get("meal_plan", [])
        chosen_store = state.get("chosen_store")
        cheapest_missing = state.get("cheapest_ingredients_info", [])
        
        # Basic check if inputs are missing
        if not chosen_store or not meal_plan:
            print("WARN (Agent 4): Missing chosen store or meal plan. Cannot generate list.")
            updated_state = state.copy()
            updated_state["shopping_list"] = {}
            updated_state["agent_outcome"] = {"status": "skipped", "reason": "Missing chosen_store or meal_plan"}
            return updated_state
        
        # Format inputs for the prompt
        meal_plan_json = json.dumps(meal_plan, indent=2, ensure_ascii=False)
        cheapest_missing_json = json.dumps(cheapest_missing, indent=2, ensure_ascii=False)
        
        # Invoke the LLM chain
        response = self.chain.invoke({
            "chosen_store": chosen_store,
            "meal_plan_json": meal_plan_json,
            "cheapest_missing_json": cheapest_missing_json
        })
        agent_output = response.content if hasattr(response, 'content') else str(response)
        
        print(f"DEBUG (Agent 4 Output): {agent_output}")
        
        # Parse and validate the output
        parsed_output, error = parse_llm_json_output(agent_output)
        
        if error:
            updated_state = state.copy()
            updated_state["shopping_list"] = {}
            updated_state["agent_outcome"] = {"status": "skipped", "reason": f"JSON parsing error: {error}"}
            return updated_state
        
        try:
            # Agent should output a dictionary
            if not isinstance(parsed_output, dict):
                # Handle edge case where it might return empty list string instead of empty object string
                if agent_output.strip() in ["[]", "null"]:
                    parsed_output = {}
                else:
                    raise ValueError("Parsed JSON is not a dictionary as expected.")
            
            # Ensure shopping list only contains the chosen store
            if len(parsed_output) > 0 and chosen_store not in parsed_output:
                # Create a new dict with only the chosen store
                store_items = []
                for items in parsed_output.values():
                    store_items.extend(items)
                parsed_output = {chosen_store: store_items}
            
            # Ensure all shopping list items have the required fields
            if chosen_store in parsed_output:
                for i, item in enumerate(parsed_output[chosen_store]):
                    if "image_url" not in item:
                        parsed_output[chosen_store][i]["image_url"] = None
                    if "notes" not in item:
                        parsed_output[chosen_store][i]["notes"] = "Deal item" if "deal" in item.get("name", "").lower() else "Staple item"
            
            print(f"DEBUG (Agent 4 Parsed): Generated shopping list with {len(parsed_output.get(chosen_store, []))} items from {chosen_store}.")
            
            # Update and return state
            updated_state = state.copy()
            updated_state["shopping_list"] = parsed_output
            updated_state["agent_outcome"] = {"shopping_list": parsed_output, "status": "success"}
            return updated_state
            
        except Exception as e:
            print(f"ERROR (Agent 4): {str(e)}")
            updated_state = state.copy()
            updated_state["shopping_list"] = {}
            updated_state["agent_outcome"] = {"status": "error", "message": f"Agent 4 Error: {str(e)}"}
            return updated_state 