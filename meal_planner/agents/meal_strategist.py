"""Agent responsible for creating meal plans based on available deals."""

import json
from textwrap import dedent
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from meal_planner.config.settings import settings
from meal_planner.models.state import MealPlannerState
from meal_planner.utils.json_helpers import parse_llm_json_output


class MealPlanningAgent:
    """Agent that creates a weekly meal plan using available deals and on-hand ingredients."""
    
    def __init__(self):
        """Initialize the Meal Planning agent."""
        self.prompt_template = dedent("""\
            You are Agent 2: The Meal Strategist. Your input is a list of grocery deals found by Agent 1 and a list of ingredients the user has on hand.
            Your goal is to choose ONE store with a good cluster of deals, create a **7-day dinner plan for 2 people** using those deals and on-hand items, and identify ingredients still missing.

            Input Deals (Full list from Agent 1):
            ```json
            {deals_json}
            ```
            Ingredients On Hand: {on_hand_list}

            Follow these steps:
            1. Analyze the deals grouped by store. Identify which single store offers the most promising combination of deals for making **dinners for a week**.
            2. Choose the single best store based on this analysis. This is EXTREMELY important as the user only wants to shop at ONE store.
            3. Create a simple meal plan with **7 dinner ideas (one for each day)** using ONLY the deals found *at the chosen store* AND items from the `Ingredients On Hand` list. 
               - DO NOT USE ANY DEALS FROM OTHER STORES. ONLY use deals from the single chosen store.
               - Prioritize using on-hand items where logical.
               - For each meal, list the main discounted ingredient(s) used (include the full deal dictionary for each deal item within a `deals_used` list)
               - Include any main on-hand ingredients used (as a list of names in `on_hand_used`)
               - Assume meals are for 2 people; it's okay if a meal naturally serves more (generates leftovers)
               - IMPORTANT: EVERY meal plan item MUST include a "notes" field - if there are likely leftovers, set it to "Likely leftovers", otherwise set it to "Serves 2"
               - Example Meal Idea Format: {{ "meal_name": "Salmon with Roasted Potatoes", "deals_used": [ {{ "id": 10118, "name": "Laksefilet Naturell 4x125g First Price", "current_price": 79.9, ... }} ], "on_hand_used": ["potatoes", "olive oil"], "notes": "Likely leftovers" }}
            4. Identify a list of common, essential ingredients required for the planned meals that are *not* present in the deals from the *chosen store* AND are *not* on the `Ingredients On Hand` list. Be generic (e.g., "salt", "pepper", "butter", "milk", "flour"). Limit to max 5-7 essential missing items.
            5. Prepare a JSON object containing three keys:
               - "chosen_store": The name of the store you selected.
               - "meal_plan": A list of **7** meal idea objects you created (incl. only deals from the chosen store).
               - "missing_ingredients": A list of the generic names of the *truly* missing essential ingredients.
            6. Respond ONLY with this JSON object. Ensure it's valid JSON.
            
            IMPORTANT: 
            - Make sure you ONLY include deals from the chosen store in each meal's "deals_used" array.
            - The store name must match EXACTLY the store names in the deals data. For example, if the chosen store is "SPAR", all deals must have "store": "SPAR".

            Example Output Format (structure is key, content may vary):
            ```json
            {{
              "chosen_store": "SPAR",
              "meal_plan": [
                {{
                  "meal_name": "Day 1: Pan-fried Torskefilet with Boiled On-Hand Potatoes",
                  "deals_used": [
                    {{ "id": 10135, "name": "Torskefilet Msc 800g First Price", "current_price": 97.9, "previous_price": 99.9, "price_drop_percentage": 2.0, "currency": "NOK", "store": "SPAR", "image_url": "https://bilder.ngdata.no/7035620084003/kmh/large.jpg" }}
                  ],
                   "on_hand_used": ["potatoes", "salt", "pepper", "butter"],
                   "notes": "Likely leftovers"
                }},
                {{
                  "meal_name": "Day 2: Leftover Torskefilet with On-Hand Salad",
                  "deals_used": [],
                  "on_hand_used": ["lettuce", "olive oil", "vinegar"],
                  "notes": "Serves 2"
                }}
                // ... (rest of the 5 days following the same structure)
              ],
              "missing_ingredients": ["milk", "flour"] // Note: onion, garlic, oil, butter etc. were on hand
            }}
            ```
        """)
        
        self.llm = ChatOpenAI(
            model=settings.llm_model, 
            temperature=settings.llm_temperature
        )
        self.chain = ChatPromptTemplate.from_template(self.prompt_template) | self.llm
        
    def run(self, state: MealPlannerState) -> MealPlannerState:
        """Run the Meal Planning agent to create a weekly meal plan.
        
        Args:
            state: The current state of the meal planning workflow.
            
        Returns:
            Updated state with meal plan, chosen store, and missing ingredients.
        """
        print("--- Running Agent 2: Meal Planning ---")
        found_deals = state.get("found_deals", [])
        on_hand_ingredients = state.get("on_hand_ingredients", [])
        
        if not found_deals:
            print("DEBUG (Agent 2): No deals found by Agent 1. Skipping.")
            updated_state = state.copy()
            updated_state["agent_outcome"] = {"status": "skipped", "reason": "No deals provided"}
            return updated_state
        
        # Format inputs for the prompt
        deals_input_json = json.dumps(found_deals, indent=2, ensure_ascii=False)
        on_hand_list_str = ", ".join(on_hand_ingredients) if on_hand_ingredients else "None"
        
        # Invoke the LLM chain
        response = self.chain.invoke({
            "deals_json": deals_input_json,
            "on_hand_list": on_hand_list_str
        })
        agent_output = response.content if hasattr(response, 'content') else str(response)
        
        print(f"DEBUG (Agent 2 Output): {agent_output}")
        
        # Parse and validate the output
        parsed_output, error = parse_llm_json_output(agent_output)
        
        if error:
            updated_state = state.copy()
            updated_state["agent_outcome"] = {"status": "error", "reason": error}
            return updated_state
        
        try:
            chosen_store = parsed_output.get("chosen_store")
            meal_plan = parsed_output.get("meal_plan", [])
            missing_ingredients = parsed_output.get("missing_ingredients", [])
            
            if not chosen_store or not isinstance(meal_plan, list) or not isinstance(missing_ingredients, list):
                raise ValueError("Parsed JSON missing required keys or has incorrect types.")
            
            # Validate meal_plan structure and ensure all required fields
            for i, meal in enumerate(meal_plan):
                if not isinstance(meal.get("deals_used"), list) or not isinstance(meal.get("on_hand_used", []), list):
                    raise ValueError("Meal plan item missing 'deals_used' list or has invalid 'on_hand_used'")
                
                # Enforce only using chosen store
                filtered_deals = [deal for deal in meal["deals_used"] if deal.get("store") == chosen_store]
                meal_plan[i]["deals_used"] = filtered_deals
                
                # Ensure notes field exists
                if "notes" not in meal:
                    meal_plan[i]["notes"] = "Serves 2"
            
            print(f"DEBUG (Agent 2 Parsed): Chose store '{chosen_store}', planned {len(meal_plan)} meals, identified {len(missing_ingredients)} missing items.")
            
            # Update and return state
            updated_state = state.copy()
            updated_state["chosen_store"] = chosen_store
            updated_state["meal_plan"] = meal_plan
            updated_state["missing_ingredients"] = missing_ingredients
            updated_state["agent_outcome"] = {"status": "success", "chosen_store": chosen_store}
            return updated_state
            
        except Exception as e:
            print(f"ERROR (Agent 2): {str(e)}")
            updated_state = state.copy()
            updated_state["agent_outcome"] = {"status": "error", "reason": f"Agent 2 Error: {str(e)}"}
            return updated_state 