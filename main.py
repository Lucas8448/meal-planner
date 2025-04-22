import os
import requests
import json
from dotenv import load_dotenv
from textwrap import dedent
from typing import Set, List, Dict, Any, Optional, TypedDict, Union
import re
import uvicorn # For running the API server
from fastapi import FastAPI # For creating the API
from fastapi.middleware.cors import CORSMiddleware # Import CORS Middleware

# Langchain Imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain.tools import tool

# LangGraph Imports
from langgraph.graph import StateGraph, END

load_dotenv()

# ─── Graph State Definition ───────────────────────────────────────────────────

class DealInfo(TypedDict):
    """Detailed information about a product deal."""
    id: int
    name: str
    current_price: float
    previous_price: Optional[float]
    price_drop_percentage: Optional[float]
    currency: str
    store: str
    image_url: Optional[str] # Added image URL

class MealPlanItem(TypedDict):
    """Structure for a single meal in the plan."""
    meal_name: str
    deals_used: List[DealInfo] # Ensure DealInfo includes image_url now
    on_hand_used: List[str]
    notes: Optional[str]

class ShoppingListItem(TypedDict):
    """Structure for an item in the final shopping list."""
    name: str
    price: float
    currency: str
    notes: Optional[str]
    image_url: Optional[str] # Added image URL

class MealPlannerState(TypedDict):
    """
    Represents the state of our meal planning graph.
    """
    initial_query: str
    on_hand_ingredients: List[str]
    search_terms: List[str]
    found_deals: List[DealInfo] # Will now include image_url
    consolidated_deals: List[DealInfo] # If used, will include image_url
    chosen_store: Optional[str]
    meal_plan: List[MealPlanItem] # Deals used within will have image_url
    missing_ingredients: List[str]
    cheapest_ingredients_info: List[Dict[str, Any]] # Keep generic for now, maybe add image later if needed
    # agent_outcome can be str for errors or the final shopping list
    shopping_list: Dict[str, List[ShoppingListItem]] # Items will now include image_url
    agent_outcome: Union[str, Dict[str, List[ShoppingListItem]]]

# ─── Constants ────────────────────────────────────────────────────────────────
KASSALAPP_BASE_URL = "https://kassal.app/api/v1"
KASSALAPP_API_KEY = os.getenv("kassalapp_api_key")
HEADERS = {"Authorization": f"Bearer {KASSALAPP_API_KEY}"}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Cache
_nearby_store_groups_cache: Optional[Set[str]] = None

# ─── Environment Variable Check ───────────────────────────────────────────────
if not KASSALAPP_API_KEY or not OPENAI_API_KEY:
    raise ValueError("API keys (KASSALAPP_API_KEY, OPENAI_API_KEY) missing in environment.")
if not all(os.getenv(k) for k in ["location_latitude", "location_longitude", "location_radius"]):
     print("Warning: Location environment variables not fully set. Nearby store filtering may fail.")

# ─── Tool Definitions ─────────────────────────────────────────────────────────

class NearbyStoresInput(BaseModel):
    pass

@tool("get_nearby_stores", args_schema=NearbyStoresInput)
def get_nearby_stores() -> str:
    """(Internal) Fetch nearby stores based on env vars as JSON string."""
    lat, lng, km = os.getenv("location_latitude"), os.getenv("location_longitude"), os.getenv("location_radius")
    if not all([lat, lng, km]): return "Error: Missing location environment variables."
    url = f"{KASSALAPP_BASE_URL}/physical-stores?size=100&lat={lat}&lng={lng}&km={km}"
    print(f"DEBUG (get_nearby_stores): Requesting URL: {url}")
    try:
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        print(f"DEBUG (get_nearby_stores): Request failed: {e}")
        return f"Error fetching nearby stores: {e}"

# --- Helper for Caching Nearby Store Groups ---
def _get_cached_nearby_store_groups() -> Set[str]:
    """Fetches/caches nearby store groups. Returns empty set on failure."""
    global _nearby_store_groups_cache
    if _nearby_store_groups_cache is not None:
        print(f"DEBUG (Cache): Using cached groups: {_nearby_store_groups_cache}")
        return _nearby_store_groups_cache

    print("DEBUG (Cache): Cache empty, fetching nearby stores...")
    nearby_stores_json_str = get_nearby_stores.invoke({})
    nearby_store_groups = set()
    if isinstance(nearby_stores_json_str, str) and not nearby_stores_json_str.startswith("Error:"):
        try:
            stores_data = json.loads(nearby_stores_json_str)
            fetched_groups = {store.get('group') for store in stores_data.get('data', []) if store.get('group')}
            if fetched_groups:
                nearby_store_groups = fetched_groups
                print(f"DEBUG (Cache): Fetched groups: {nearby_store_groups}")
            else:
                print("DEBUG (Cache): No groups found in response.")
        except json.JSONDecodeError:
            print("DEBUG (Cache): Failed to decode JSON from get_nearby_stores.")
    else:
        print(f"DEBUG (Cache): get_nearby_stores error/type: {nearby_stores_json_str}")

    _nearby_store_groups_cache = nearby_store_groups # Cache result (even empty set)
    return nearby_store_groups

# --- Main Search Tool ---
class SearchProductsInput(BaseModel):
    search: str = Field(description="The Norwegian search term for products.")
    filter_by_price_drop: bool = Field(default=True, description="Whether to filter results to only include products with a recent price drop.")

@tool("search_products", args_schema=SearchProductsInput)
def search_products(search: str, filter_by_price_drop: bool = True) -> List[DealInfo] | str:
    """Search products by term, optionally filter by nearby stores (cached) & price drops.
    If filter_by_price_drop is True (default), returns simplified list of products with actual price drops:
    [{'id', 'name', 'current_price', 'previous_price', 'price_drop_percentage', 'currency', 'store'}]
    If filter_by_price_drop is False, returns simplified list of found products:
    [{'id', 'name', 'current_price', 'currency', 'store'}]"""
    if not isinstance(search, str) or not search.strip():
        return "Error: search term must be a non-empty string."

    nearby_store_groups = _get_cached_nearby_store_groups()
    url = f"{KASSALAPP_BASE_URL}/products?search={search}&size=100" # Increased size earlier
    print(f"DEBUG (search_products): Requesting URL: {url} | Nearby Groups: {nearby_store_groups or 'None/Empty'} | Filter Drops: {filter_by_price_drop}")

    try:
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        products_raw = data.get("data")

        if not isinstance(products_raw, list):
            return "Error: Unexpected API response format (missing 'data' list)."

        print(f"DEBUG (search_products): Received {len(products_raw)} raw products for '{search}'.")

        final_products = []
        for i, prod in enumerate(products_raw):
            store_obj = prod.get("store")
            store_code = store_obj.get("code") if store_obj else None
            store_name = store_obj.get("name") if store_obj else "N/A"
            current_price = prod.get("current_price")
            prod_id = prod.get("id")
            prod_name = prod.get("name")

            # 1. Basic Check (ID, Name, Price)
            if not (prod_id and prod_name and current_price is not None):
                continue

            # 2. Nearby Store Check (if groups exist)
            if nearby_store_groups and store_code not in nearby_store_groups:
                continue

            # --- Conditional Price Drop Filtering --- #
            if filter_by_price_drop:
                # --- Inline History Processing --- (Only needed if filtering by drop)
                raw_history = prod.get("price_history", [])
                processed_history = []
                if raw_history:
                    try:
                        sorted_history = sorted(raw_history, key=lambda x: x.get('date', ''), reverse=True)
                        for entry in sorted_history[:10]:
                            price_str = entry.get('price')
                            if price_str is not None:
                                try: processed_history.append(float(price_str))
                                except (ValueError, TypeError): pass
                    except Exception as e:
                        print(f"DEBUG (search_products): Error processing history for prod {prod_id}: {e}")
                        processed_history = []
                # --- End Inline History Processing ---

                if not processed_history: # Need history to check for drops
                    continue

                # 3. Find Previous Price
                previous_price = None
                for price in processed_history:
                    if price != current_price:
                        previous_price = price
                        break

                # 4. Price Drop Check & Calculation
                if previous_price is not None and previous_price > current_price:
                    try:
                        price_drop_percentage = round(((previous_price - current_price) / previous_price) * 100, 2)
                        if price_drop_percentage > 0: # Double check it's a drop
                            # Add product with drop details
                            final_products.append({
                                "id": prod_id,
                                "name": prod_name,
                                "current_price": current_price,
                                "previous_price": previous_price,
                                "price_drop_percentage": price_drop_percentage,
                                "currency": "NOK",
                                "store": store_name,
                                "image_url": prod.get("image"),
                            })
                    except ZeroDivisionError:
                        continue # Skip if previous price was 0
                # If no price drop, this product is skipped when filter_by_price_drop is True

            else: # filter_by_price_drop is False - just add basic details
                final_products.append({
                    "id": prod_id,
                    "name": prod_name,
                    "current_price": current_price,
                    "currency": "NOK",
                    "store": store_name,
                    "image_url": prod.get("image"),
                })
            # --- End Conditional Filtering --- #

        print(f"DEBUG (search_products): Returning {len(final_products)} products for '{search}' (Filter Drops: {filter_by_price_drop}).")
        return final_products

    except requests.exceptions.HTTPError as http_err:
        status = http_err.response.status_code
        body_snippet = http_err.response.text[:200] + "..." if http_err.response.text else ""
        return f"Error searching products: HTTP {status}. Body: {body_snippet}"
    except requests.exceptions.RequestException as e:
        return f"Error searching products: Request Exception: {e}"
    except json.JSONDecodeError:
        return "Error: Failed to decode JSON response from product search API."
    except Exception as e:
        print(f"ERROR (search_products): Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: An unexpected error occurred: {e}"


# --- Get Product Details Tool ---
class ProductDetailsInput(BaseModel):
    product_id: int | str = Field(description="The EAN or internal ID of the product.")

@tool("get_product_details", args_schema=ProductDetailsInput)
def get_product_details(product_id: int | str) -> Dict[str, Any] | str:
    """Fetch detailed information for a single product by its EAN or ID."""
    if not product_id:
        return "Error: product_id must be provided."

    url = f"{KASSALAPP_BASE_URL}/products/{product_id}"
    print(f"DEBUG (get_product_details): Requesting URL: {url}")

    try:
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        product_data = data.get("data")

        if not product_data:
             return "Error: Product data not found in API response."

        # Select and potentially simplify the returned fields
        relevant_details = {
            "id": product_data.get("id"),
            "ean": product_data.get("ean"),
            "name": product_data.get("name"),
            "description": product_data.get("description"),
            "brand": product_data.get("brand"),
            "vendor": product_data.get("vendor"),
            "current_price": product_data.get("current_price"),
            "price_per_unit": product_data.get("price_per_unit"),
            "unit_measure_name": product_data.get("unit_measure_name"),
            "category": product_data.get("category_name"),
            "store": product_data.get("store", {}).get("name"),
            "store_code": product_data.get("store", {}).get("code"),
            "image_url": product_data.get("image"),
            "url": product_data.get("url"),
            "last_updated": product_data.get("updated"),
        }
        return {k: v for k, v in relevant_details.items() if v is not None} # Clean nulls

    except requests.exceptions.HTTPError as http_err:
        status = http_err.response.status_code
        if status == 404:
            return f"Error: Product with ID/EAN '{product_id}' not found."
        body_snippet = http_err.response.text[:200] + "..." if http_err.response.text else ""
        return f"Error fetching product details: HTTP {status}. Body: {body_snippet}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching product details: Request Exception: {e}"
    except json.JSONDecodeError:
        return "Error: Failed to decode JSON response from product details API."
    except Exception as e:
        print(f"ERROR (get_product_details): Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: An unexpected error occurred: {e}"


# ─── Agent Definitions ────────────────────────────────────────────────────────

# --- Agent 1: Deal Hunter ---

deal_hunter_prompt = ChatPromptTemplate.from_messages([
    ("system", dedent("""\
        You are Agent 1: The Deal Hunter. Your goal is to find Norwegian grocery products with recent price drops suitable for common dinners.

        Follow these steps strictly:
        1. Based on the user's initial query in the 'input', generate a diverse list of 15-20 specific Norwegian dinner-related search terms (e.g., 'svinekoteletter', 'torsk', 'kyllingfilet', 'laksefilet', 'kjøttdeig', 'gulrot', 'potet', 'løk', 'tomat', 'pasta', 'ris', 'laks', 'kylling', 'brokkoli', 'blomkål'). Prioritize common ingredients suitable for multiple meals.
        2. Immediately call the `search_products` tool sequentially for each generated term.
        3. The `search_products` tool returns ONLY products with an actual price drop (current price < previous price). The fields returned are: `id`, `name`, `current_price`, `previous_price`, `price_drop_percentage`, `currency`, `store`.
        4. Collect ALL valid results returned by the tool calls across all search terms. Do not filter further yet.
        5. Prepare a JSON object containing two keys:
           - "search_terms": A list of the search terms you generated and used.
           - "found_deals": A list of all the product deals found by the tool calls (the list returned by `search_products`).
        6. Respond ONLY with this JSON object. Do not add any explanations or conversational text.

        Example Output Format:
        ```json
        {{
          "search_terms": ["svinekoteletter", "torsk", "kyllingfilet", ...],
          "found_deals": [
            {{"id": 123, "name": "Gilde Svinekoteletter", "current_price": 45.5, ...}},
            {{"id": 456, "name": "Torskefilet", "current_price": 89.0, ...}},
            ...
          ]
        }}
        ```
    """)),
    ("user", "{input}"), # This will be the state['initial_query']
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Use the specific tools needed by this agent
deal_hunter_tools = [search_products]
deal_hunter_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0) # Lower temp for more deterministic JSON output
deal_hunter_agent = create_openai_functions_agent(deal_hunter_llm, deal_hunter_tools, deal_hunter_prompt)
deal_hunter_executor = AgentExecutor(agent=deal_hunter_agent, tools=deal_hunter_tools, verbose=False, max_iterations=25)

# --- Agent 2: Meal Strategist ---

# Reusing the LLM from Agent 1 for simplicity
meal_strategist_llm = deal_hunter_llm # Or define ChatOpenAI(...) again if different settings needed

meal_strategist_prompt_template = dedent("""\
    You are Agent 2: The Meal Strategist. Your input is a list of grocery deals found by Agent 1 and a list of ingredients the user has on hand.
    Your goal is to choose ONE store with a good cluster of deals, create a **7-day dinner plan for 2 people** using those deals and on-hand items, and identify ingredients still missing.

    Input Deals (Full list from Agent 1):
    ```json
    {deals_json}
    ```
    Ingredients On Hand: {on_hand_list}

    Follow these steps:
    1. Analyze the deals grouped by store. Identify which single store offers the most promising combination of deals for making **dinners for a week**.
    2. Choose the single best store based on this analysis.
    3. Create a simple meal plan with **7 dinner ideas (one for each day)** using ONLY the deals found *at the chosen store* AND items from the `Ingredients On Hand` list. Prioritize using on-hand items where logical. For each meal, list the main discounted ingredient(s) used (include the **full deal dictionary** for each deal item used in that meal within a `deals_used` list) and any main on-hand ingredients used (as a list of names in `on_hand_used`).
       - Assume meals are for 2 people; it's okay if a meal naturally serves more (generates leftovers) - if so, you can optionally add a "notes": "Likely leftovers" field to that meal object.
       - Example Meal Idea Format: {{ "meal_name": "Salmon with Roasted Potatoes", "deals_used": [ {{ "id": 10118, "name": "Laksefilet Naturell 4x125g First Price", "current_price": 79.9, ... }} ], "on_hand_used": ["potatoes", "olive oil"], "notes": "Likely leftovers" }}
    4. Identify a list of common, essential ingredients required for the planned meals that are *not* present in the deals from the *chosen store* AND are *not* on the `Ingredients On Hand` list. Be generic (e.g., "salt", "pepper", "butter", "milk", "flour"). Limit to max 5-7 essential missing items.
    5. Prepare a JSON object containing three keys:
       - "chosen_store": The name of the store you selected.
       - "meal_plan": A list of **7** meal idea objects you created (incl. `deals_used` dicts and `on_hand_used` list).
       - "missing_ingredients": A list of the generic names of the *truly* missing essential ingredients (after checking deals and on-hand list).
    6. Respond ONLY with this JSON object. Ensure it's valid JSON.

    Example Output Format (structure is key, content may vary):
    ```json
    {{
      "chosen_store": "SPAR",
      "meal_plan": [
        {{
          "meal_name": "Day 1: Pan-fried Torskefilet with Boiled On-Hand Potatoes",
          "deals_used": [
            {{ "id": 10135, "name": "Torskefilet Msc 800g First Price", ... }}
          ],
           "on_hand_used": ["potatoes", "salt", "pepper", "butter"]
        }},
        {{
          "meal_name": "Day 2: Leftover Torskefilet with On-Hand Salad",
          "deals_used": [],
          "on_hand_used": ["lettuce", "olive oil", "vinegar"]
        }}
        // ... (rest of the 5 days following the same structure)
      ],
      "missing_ingredients": ["milk", "flour"] // Note: onion, garlic, oil, butter etc. were on hand
    }}
    ```
""")

# Chain for Agent 2 (needs updating to accept on_hand_list)
# We need to adjust how the chain is invoked later in the node function.
# Let's keep the chain definition simple for now.
meal_strategist_chain = (
    ChatPromptTemplate.from_template(meal_strategist_prompt_template) | meal_strategist_llm
)

# --- Agent 3: Bargain Scout ---

# Tools needed by this agent
bargain_scout_tools = [search_products, get_product_details]
bargain_scout_llm = deal_hunter_llm # Reusing LLM

bargain_scout_prompt_template_str = dedent("""\
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

    Example Output Format:
    ```json
    [
      {{
        "ingredient_name": "onion",
        "product_id": 654,
        "product_name": "Løk 1kg",
        "store": "Rema 1000",
        "current_price": 18.90,
        "unit": "kg"
      }},
      {{
        "ingredient_name": "garlic",
        "product_id": 5122,
        "product_name": "Hvitløk pk", // Note: Using 'pk' as unit might be necessary
        "store": "Joker",
        "current_price": 19.90,
        "unit": "stk" // Or could be null if API doesn't provide
      }},
      {{
        "ingredient_name": "olive oil",
        "product_id": 10048,
        "product_name": "Extra Virgin Olivenolje 500ml Eldorado",
        "store": "KIWI",
        "current_price": 65.50,
        "unit": "l" // Unit might be 'l' even if name says 500ml
      }}
      // ... other found ingredients
    ]
    ```
""")

# Corrected prompt definition using from_messages
bargain_scout_prompt = ChatPromptTemplate.from_messages([
    ("system", bargain_scout_prompt_template_str),
    ("user", "Find the cheapest options for these missing ingredients: {missing_ingredients_list}"), # User input trigger
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Agent Executor for Agent 3
# Use the corrected prompt object `bargain_scout_prompt`
bargain_scout_agent = create_openai_functions_agent(bargain_scout_llm, bargain_scout_tools, bargain_scout_prompt)
# Needs high iterations due to potential loop (search + multiple details per ingredient)
bargain_scout_executor = AgentExecutor(agent=bargain_scout_agent, tools=bargain_scout_tools, verbose=False, max_iterations=40)

# --- Agent 4: List Consolidator ---

list_consolidator_llm = deal_hunter_llm # Reusing LLM

list_consolidator_prompt_template = dedent("""\
    You are Agent 4: The List Consolidator. Your goal is to create a final shopping list organized by store.

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
    1. Extract all the **deal items** directly from the `deals_used` lists within the `meal_plan_json`. These items are guaranteed to be from the `chosen_store`.
    2. Collect the items from the `cheapest_missing_json` list. These represent standard options found for the missing ingredients needed to supplement the meal plan.
    3. Combine the items from step 1 (deals used in the plan) and step 2 (found missing ingredients).
    4. Organize the combined items into a final shopping list grouped by store. The output should be a JSON object where keys are store names, and values are lists of items to buy at that store.
    5. Each item in the store list should be a dictionary containing at least:
       - `name`: Product name.
       - `price`: The `current_price` of the item.
       - `currency`: Currency code (e.g., "NOK").
       - `notes`: Optional field. Add "Deal item" for items from step 1. For items from step 2, add "Staple item for [ingredient_name]". Add "Likely leftovers" if noted in the meal plan for a deal item (append if other notes exist, e.g., "Deal item; Likely leftovers").

    6. Respond ONLY with the final JSON shopping list object. Do not add explanations. If the combined list is empty, respond with `{{}}`.

    Example Output Format:
    ```json
    {{
      "SPAR": [
        {{ "name": "Torskefilet Msc 800g First Price", "price": 97.9, "currency": "NOK", "notes": "Deal item" }},
        {{ "name": "Poteter Kokte u/Skall 375g Grønn&Frisk", "price": 21.9, "currency": "NOK", "notes": "Deal item" }},
        {{ "name": "Pasta Carbonara 600g Findus", "price": 66.9, "currency": "NOK", "notes": "Deal item; Likely leftovers" }}
        // ... other SPAR deal items ...
      ],
      "Rema 1000": [
         {{ "name": "Løk 1kg", "price": 18.90, "currency": "NOK", "notes": "Staple item for onion" }}
      ],
      "Joker": [
         {{ "name": "Hvitløk pk", "price": 19.90, "currency": "NOK", "notes": "Staple item for garlic" }}
      ],
       "KIWI": [
         {{ "name": "Extra Virgin Olivenolje 500ml Eldorado", "price": 65.50, "currency": "NOK", "notes": "Staple item for olive oil" }}
      ]
      // ... other stores and staple items ...
    }}
    ```
""")

list_consolidator_chain = (
    ChatPromptTemplate.from_template(list_consolidator_prompt_template) | list_consolidator_llm
)


# ─── Graph Node Functions ─────────────────────────────────────────────────────

def run_deal_hunter(state: MealPlannerState) -> MealPlannerState:
    """Runs the Deal Hunter agent to find products with price drops."""
    print("--- Running Agent 1: Deal Hunter ---")
    initial_query = state['initial_query']
    # Ensure agent_scratchpad is handled correctly if needed, though create_openai_functions_agent often manages it implicitly
    # For stateful graphs, managing message history might be needed if the agent needs context from previous turns within the same node execution (unlikely here).
    result = deal_hunter_executor.invoke({"input": initial_query})
    agent_output = result.get("output", "{}") # Default to empty JSON string

    print(f"DEBUG (Agent 1 Output): {agent_output}")

    # --- Strip Markdown Code Block --- #
    clean_agent_output = agent_output.strip()
    if clean_agent_output.startswith("```json"):
        clean_agent_output = clean_agent_output[7:] # Remove ```json\n
    if clean_agent_output.endswith("```"):
        clean_agent_output = clean_agent_output[:-3] # Remove ```
    clean_agent_output = clean_agent_output.strip() # Remove any extra whitespace
    # --- End Strip Markdown Code Block --- #

    try:
        parsed_output = json.loads(clean_agent_output)
        search_terms = parsed_output.get("search_terms", [])
        found_deals = parsed_output.get("found_deals", [])

        if not isinstance(search_terms, list) or not isinstance(found_deals, list):
             raise ValueError("Parsed JSON does not contain lists for 'search_terms' or 'found_deals'")

        print(f"DEBUG (Agent 1 Parsed): Found {len(found_deals)} deals using {len(search_terms)} terms.")

        # Update state - use .update() for TypedDict compatibility if needed, or direct assignment
        # return {**state, "search_terms": search_terms, "found_deals": found_deals, "agent_outcome": parsed_output}
        updated_state = state.copy() # Avoid modifying input state directly if running concurrently
        updated_state["search_terms"] = search_terms
        updated_state["found_deals"] = found_deals
        updated_state["agent_outcome"] = parsed_output # Store raw agent output if needed
        return updated_state

    except json.JSONDecodeError as e:
        print(f"ERROR (Agent 1): Failed to parse JSON output: {e}\nOutput was: {agent_output}")
        # Decide how to handle errors - stop graph, retry, return default state?
        # For now, return state with empty lists and the error outcome
        # return {**state, "search_terms": [], "found_deals": [], "agent_outcome": {"error": f"JSON Parse Error: {e}"}}
        updated_state = state.copy()
        updated_state["search_terms"] = []
        updated_state["found_deals"] = []
        updated_state["agent_outcome"] = {"error": f"JSON Parse Error: {e}"}
        return updated_state
    except ValueError as e:
        print(f"ERROR (Agent 1): Invalid data types in parsed JSON: {e}\nOutput was: {agent_output}")
        # return {**state, "search_terms": [], "found_deals": [], "agent_outcome": {"error": f"Data Type Error: {e}"}}
        updated_state = state.copy()
        updated_state["search_terms"] = []
        updated_state["found_deals"] = []
        updated_state["agent_outcome"] = {"error": f"Data Type Error: {e}"}
        return updated_state

def run_meal_strategist(state: MealPlannerState) -> MealPlannerState:
    """Runs the Meal Strategist to choose a store, plan meals, and identify missing items, considering on-hand ingredients."""
    print("--- Running Agent 2: Meal Strategist ---")
    found_deals = state.get("found_deals", [])
    on_hand_ingredients = state.get("on_hand_ingredients", []) # Get on-hand list from state

    if not found_deals:
        print("DEBUG (Agent 2): No deals found by Agent 1. Skipping.")
        # Optionally update state to reflect skipping or return as is
        updated_state = state.copy()
        updated_state["agent_outcome"] = {"status": "skipped", "reason": "No deals provided"}
        return updated_state # Or potentially route to END directly if possible

    # Format inputs for the prompt
    deals_input_json = json.dumps(found_deals, indent=2, ensure_ascii=False)
    on_hand_list_str = ", ".join(on_hand_ingredients) if on_hand_ingredients else "None"

    # Invoke the LLM chain with both inputs
    response = meal_strategist_chain.invoke({
        "deals_json": deals_input_json,
        "on_hand_list": on_hand_list_str
    })
    agent_output = response.content if hasattr(response, 'content') else str(response)

    print(f"DEBUG (Agent 2 Output): {agent_output}")

    # --- Strip Markdown Code Block --- #
    clean_agent_output = agent_output.strip()
    if clean_agent_output.startswith("```json"):
        clean_agent_output = clean_agent_output[7:].strip()
    elif clean_agent_output.startswith("```"): # Handle ``` without 'json' too
         clean_agent_output = clean_agent_output[3:].strip()
    if clean_agent_output.endswith("```"):
        clean_agent_output = clean_agent_output[:-3].strip()
    # --- End Strip Markdown Code Block --- #

    try:
        parsed_output = json.loads(clean_agent_output)
        chosen_store = parsed_output.get("chosen_store")
        meal_plan = parsed_output.get("meal_plan", [])
        missing_ingredients = parsed_output.get("missing_ingredients", [])

        if not chosen_store or not isinstance(meal_plan, list) or not isinstance(missing_ingredients, list):
             raise ValueError("Parsed JSON missing required keys or has incorrect types.")

         # Added validation for new 'on_hand_used' key in meal_plan items
        for meal in meal_plan:
            if not isinstance(meal.get("deals_used"), list) or not isinstance(meal.get("on_hand_used", []), list):
                 raise ValueError("Meal plan item missing 'deals_used' list or has invalid 'on_hand_used'")

        print(f"DEBUG (Agent 2 Parsed): Chose store '{chosen_store}', planned {len(meal_plan)} meals, identified {len(missing_ingredients)} missing items.")

        updated_state = state.copy()
        updated_state["chosen_store"] = chosen_store
        updated_state["meal_plan"] = meal_plan
        updated_state["missing_ingredients"] = missing_ingredients
        updated_state["agent_outcome"] = parsed_output # Store Agent 2's output
        return updated_state

    except json.JSONDecodeError as e:
        print(f"ERROR (Agent 2): Failed to parse JSON output: {e}\nOutput was: {clean_agent_output}")
        updated_state = state.copy()
        updated_state["agent_outcome"] = {"error": f"Agent 2 JSON Parse Error: {e}", "raw_output": agent_output}
        # How to handle? Maybe stop or try to recover? For now, just log and pass state.
        return updated_state
    except ValueError as e:
        print(f"ERROR (Agent 2): Invalid data in parsed JSON: {e}\nOutput was: {clean_agent_output}")
        updated_state = state.copy()
        updated_state["agent_outcome"] = {"error": f"Agent 2 Data Error: {e}", "raw_output": agent_output}
        return updated_state

def run_bargain_scout(state: MealPlannerState) -> MealPlannerState:
    """Runs the Bargain Scout agent to find the cheapest options for missing ingredients."""
    print("--- Running Agent 3: Bargain Scout ---")
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
    response = bargain_scout_executor.invoke({"missing_ingredients_list": missing_ingredients_str})
    agent_output = response.get("output", "[]") # Default to empty list string

    print(f"DEBUG (Agent 3 Output): {agent_output}")

    # --- Strip Markdown Code Block --- #
    clean_agent_output = agent_output.strip()
    # Handle potential ```json ... ``` or just ``` ... ```
    if clean_agent_output.startswith("```"):
        # Remove prefix (```json or ```) and newline
        clean_agent_output = re.sub(r"^```(?:json)?\s*", "", clean_agent_output)
    if clean_agent_output.endswith("```"):
        clean_agent_output = clean_agent_output[:-3].strip()
    # --- End Strip Markdown Code Block --- #

    try:
        # Agent should output a list directly
        parsed_output = json.loads(clean_agent_output)
        if not isinstance(parsed_output, list):
            raise ValueError("Parsed JSON is not a list as expected.")

        # Basic validation of list items (optional but recommended)
        validated_info = []
        required_keys = {"ingredient_name", "product_id", "product_name", "store", "current_price", "unit"}
        for item in parsed_output:
            if isinstance(item, dict) and required_keys.issubset(item.keys()):
                validated_info.append(item)
            else:
                print(f"WARN (Agent 3): Skipping invalid item in output: {item}")

        print(f"DEBUG (Agent 3 Parsed): Found cheapest options for {len(validated_info)} ingredients.")

        updated_state = state.copy()
        updated_state["cheapest_ingredients_info"] = validated_info
        updated_state["agent_outcome"] = validated_info # Store Agent 3's processed output
        return updated_state

    except json.JSONDecodeError as e:
        print(f"ERROR (Agent 3): Failed to parse JSON list output: {e}\nOutput was: {clean_agent_output}")
        updated_state = state.copy()
        updated_state["cheapest_ingredients_info"] = []
        updated_state["agent_outcome"] = {"error": f"Agent 3 JSON Parse Error: {e}", "raw_output": agent_output}
        return updated_state
    except ValueError as e:
        print(f"ERROR (Agent 3): Invalid data in parsed JSON: {e}\nOutput was: {clean_agent_output}")
        updated_state = state.copy()
        updated_state["cheapest_ingredients_info"] = []
        updated_state["agent_outcome"] = {"error": f"Agent 3 Data Error: {e}", "raw_output": agent_output}
        return updated_state

def run_list_consolidator(state: MealPlannerState) -> MealPlannerState:
    """Runs the List Consolidator agent to create the final shopping list."""
    print("--- Running Agent 4: List Consolidator ---")

    meal_plan = state.get("meal_plan", [])
    all_deals = state.get("found_deals", []) # Deals found by Agent 1
    chosen_store = state.get("chosen_store")
    cheapest_missing = state.get("cheapest_ingredients_info", [])

    # Basic check if inputs are missing
    if not chosen_store or not meal_plan:
         print("WARN (Agent 4): Missing chosen store or meal plan. Cannot generate list.")
         updated_state = state.copy()
         updated_state["shopping_list"] = {}
         updated_state["agent_outcome"] = {"status": "skipped", "reason": "Missing chosen_store or meal_plan"}
         return updated_state

    # Format inputs for prompt
    meal_plan_json = json.dumps(meal_plan, indent=2, ensure_ascii=False)
    all_deals_json = json.dumps(all_deals, indent=2, ensure_ascii=False)
    cheapest_missing_json = json.dumps(cheapest_missing, indent=2, ensure_ascii=False)

    # Invoke the LLM chain
    response = list_consolidator_chain.invoke({
        "chosen_store": chosen_store,
        "meal_plan_json": meal_plan_json,
        "all_deals_json": all_deals_json,
        "cheapest_missing_json": cheapest_missing_json,
    })
    agent_output = response.content if hasattr(response, 'content') else str(response)

    print(f"DEBUG (Agent 4 Output): {agent_output}")

    # --- Strip Markdown Code Block --- #
    clean_agent_output = agent_output.strip()
    # Handle potential ```json ... ``` or just ``` ... ```
    if clean_agent_output.startswith("```"):
        clean_agent_output = re.sub(r"^```(?:json)?\s*", "", clean_agent_output)
    if clean_agent_output.endswith("```"):
        clean_agent_output = clean_agent_output[:-3].strip()
    # --- End Strip Markdown Code Block --- #

    try:
        # Agent should output a dictionary (shopping list)
        parsed_output = json.loads(clean_agent_output)
        if not isinstance(parsed_output, dict):
            # Handle edge case where it might return empty list string instead of empty object string
            if clean_agent_output == "[]":
                parsed_output = {}
            else:
                raise ValueError("Parsed JSON is not a dictionary as expected.")

        # Optional: Add validation for the shopping list structure
        print(f"DEBUG (Agent 4 Parsed): Generated shopping list with {len(parsed_output)} stores.")

        updated_state = state.copy()
        updated_state["shopping_list"] = parsed_output
        updated_state["agent_outcome"] = parsed_output # Store Agent 4's output
        return updated_state

    except json.JSONDecodeError as e:
        print(f"ERROR (Agent 4): Failed to parse JSON object output: {e}\nOutput was: {clean_agent_output}")
        updated_state = state.copy()
        updated_state["shopping_list"] = {} # Set empty list on error
        updated_state["agent_outcome"] = {"error": f"Agent 4 JSON Parse Error: {e}", "raw_output": agent_output}
        return updated_state
    except ValueError as e:
        print(f"ERROR (Agent 4): Invalid data in parsed JSON: {e}\nOutput was: {clean_agent_output}")
        updated_state = state.copy()
        updated_state["shopping_list"] = {}
        updated_state["agent_outcome"] = {"error": f"Agent 4 Data Error: {e}", "raw_output": agent_output}
        return updated_state


# ─── Graph Definition & Compilation ─────────────────────────────────────────

workflow = StateGraph(MealPlannerState)

# Add nodes
workflow.add_node("deal_hunter", run_deal_hunter)
workflow.add_node("meal_strategist", run_meal_strategist)
workflow.add_node("bargain_scout", run_bargain_scout)
workflow.add_node("list_consolidator", run_list_consolidator)

# Define edges
workflow.set_entry_point("deal_hunter")
workflow.add_edge("deal_hunter", "meal_strategist")
workflow.add_edge("meal_strategist", "bargain_scout")
workflow.add_edge("bargain_scout", "list_consolidator")
workflow.add_edge("list_consolidator", END)

# Compile the graph - happens once when the script starts
app = workflow.compile()

# ─── API Definition ───────────────────────────────────────────────────────────

api_app = FastAPI(
    title="Meal Planner Agent API",
    description="API to run the multi-agent meal planning process based on grocery deals.",
    version="0.1.0",
)

# Configure CORS
origins = [
    "*", # Allow all origins for local development
    # Add specific origins for production, e.g.:
    # "http://your-frontend-domain.com",
    # "https://your-frontend-domain.com",
]

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of origins allowed (or "*" for all)
    allow_credentials=True, # Allow cookies
    allow_methods=["*"],    # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],    # Allow all headers
)

# Define Pydantic models for API request/response
class PlanRequest(BaseModel):
    query: str = "Find common dinner ingredients in Norway with recent price drops to help plan meals."
    on_hand_ingredients: Optional[List[str]] = Field(default_factory=list, description="Optional list of ingredient names user already has.")

# Using MealPlannerState directly as response model implies all fields are returned
# If we wanted to return a subset, we'd define a specific ResponseModel

# Change to POST to accept request body
@api_app.post("/plan-meals", response_model=MealPlannerState)
async def run_meal_plan_endpoint(request: PlanRequest) -> MealPlannerState:
    """Runs the full meal planning agent graph based on an initial query and optional on-hand ingredients."""
    print(f"--- API Request Received: /plan-meals ---")
    print(f"Query: {request.query}")
    print(f"On Hand: {request.on_hand_ingredients}")

    initial_state: MealPlannerState = {
        "initial_query": request.query,
        "on_hand_ingredients": request.on_hand_ingredients or [], # Use provided list or empty list
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

    # Invoke the compiled LangGraph app (synchronously for now)
    # For production, consider running invoke in a threadpool if it's CPU-bound
    # or refactoring tools/nodes to be async for ainvoke.
    try:
        # Note: Directly calling app.invoke() in an async endpoint
        # will block the event loop. For long-running graphs, use
        # asyncio.to_thread or refactor the graph to be async.
        # For simplicity here, we'll keep it sync.
        final_state = app.invoke(initial_state)
        print("--- API Request Completed Successfully ---")
        return final_state
    except Exception as e:
        print(f"ERROR during graph invocation: {e}")
        # Consider returning an HTTP error response
        # from fastapi import HTTPException
        # raise HTTPException(status_code=500, detail=str(e))
        # For now, return a basic error structure within the state
        error_state = initial_state.copy()
        error_state["agent_outcome"] = {"error": f"Graph execution failed: {e}"}
        return error_state

# ─── Server Execution ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("## Starting Meal Planner API Server...")
    # Use uvicorn to run the FastAPI app
    # "main:api_app" tells uvicorn to look for the `api_app` object in the `main.py` file.
    # reload=True is helpful for development; remove for production.
    uvicorn.run("main:api_app", host="0.0.0.0", port=8000, reload=True)