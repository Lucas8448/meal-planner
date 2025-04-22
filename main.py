import os
import requests
import json
from dotenv import load_dotenv
from textwrap import dedent
from typing import Set, List, Dict, Any, Optional

# Langchain Imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain.tools import tool

load_dotenv()

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

@tool("search_products", args_schema=SearchProductsInput)
def search_products(search: str) -> List[Dict[str, Any]] | str:
    """Search products by term, filter by nearby stores (cached) & price drops.
    Returns simplified list of products with actual price drops:
    [{'id', 'name', 'current_price', 'previous_price', 'price_drop_percentage', 'currency', 'store'}]"""
    if not isinstance(search, str) or not search.strip():
        return "Error: search term must be a non-empty string."

    nearby_store_groups = _get_cached_nearby_store_groups()
    url = f"{KASSALAPP_BASE_URL}/products?search={search}&size=40"
    print(f"DEBUG (search_products): Requesting URL: {url} | Nearby Groups: {nearby_store_groups or 'None/Empty'}")

    try:
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        products_raw = data.get("data")

        if not isinstance(products_raw, list):
            return "Error: Unexpected API response format (missing 'data' list)."

        print(f"DEBUG (search_products): Received {len(products_raw)} raw products for '{search}'.")

        final_products_with_drops = []
        for i, prod in enumerate(products_raw):
            # --- Inline History Processing ---
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
                    print(f"DEBUG (search_products): Error processing history for prod {prod.get('id')}: {e}")
                    processed_history = []
            # --- End Inline History Processing ---

            store_obj = prod.get("store")
            store_code = store_obj.get("code") if store_obj else None
            store_name = store_obj.get("name") if store_obj else "N/A"
            current_price = prod.get("current_price")
            prod_id = prod.get("id")
            prod_name = prod.get("name")

            # Debugging Log (optional, can be removed later)
            # print(f"  [DEBUG {search} #{i+1}] ID: {prod_id}, Name: {prod_name}, Price: {current_price}, Store: {store_name} ({store_code}), History: {processed_history}")

            # --- Filtering and Calculation ---
            # 1. Basic Check
            if not (prod_id and prod_name and current_price is not None and processed_history):
                continue

            # 2. Nearby Store Check (if groups exist)
            if nearby_store_groups and store_code not in nearby_store_groups:
                continue

            # 3. Find Previous Price and Calculate Drop
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
                        # Add product to final list ONLY if a drop occurred
                        final_products_with_drops.append({
                            "id": prod_id,
                            "name": prod_name,
                            "current_price": current_price,
                            "previous_price": previous_price,
                            "price_drop_percentage": price_drop_percentage,
                            "currency": "NOK",
                            "store": store_name,
                        })
                except ZeroDivisionError:
                    continue # Skip if previous price was 0

            # --- End Filtering and Calculation ---

        print(f"DEBUG (search_products): Found {len(final_products_with_drops)} products with price drops for '{search}'.")
        return final_products_with_drops

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


# ─── Langchain Agent Setup ─────────────────────────────────────────────────────
tools = [search_products] # Only need the main tool
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=1)

# Slightly refined prompt
prompt_template = ChatPromptTemplate.from_messages([
    ("system", dedent("""\
        You are an Insightful Shopaholic finding Norwegian dinner deals with price drops.

        Follow these steps strictly:
        1. Generate a diverse list of 10-15 specific Norwegian dinner-related search terms (e.g., 'svinekoteletter', 'torsk', 'lasagne', 'linsegryte').
        2. Immediately call `search_products` sequentially for each generated term.
        3. The `search_products` tool internally filters products and returns ONLY those with an actual price drop (current price < previous price) and includes calculated fields. The returned list format for each product is: `id`, `name`, `current_price` (float), `previous_price` (float), `price_drop_percentage` (float, positive), `currency`, `store` (name).
        4. Collect all results from the tool calls.
        5. Sort the collected products by `price_drop_percentage` (descending).
        6. Prepare a JSON list (top-level element) of up to 30 products from the sorted list, keeping the fields returned by the tool: `id`, `name`, `price` (current_price), `currency`, `store`, `previous_price`, `price_drop_percentage`.
        7. If the final sorted list is empty (no products with drops found across all searches), respond ONLY with: "No dinner products with recent price drops found for the searched terms."
        Output Format: Respond ONLY with the final JSON list or the 'not found' message. No explanations.
    """)),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False) # Keep verbose=False for production


# ─── Execution ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("## Langchain Agent Kicking Off!")
    initial_input = "Find the single dinner product in Norway with the largest recent price drop."
    result = agent_executor.invoke({"input": initial_input})

    print("\n## Agent Run Completed")
    final_output = result.get("output", "No output key found in result.")
    print("\nFinal Output:")
    print(final_output)

    # Attempt to pretty-print if JSON
    try:
        parsed_json = json.loads(final_output)
        print("\nFormatted JSON Output:")
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
    except (json.JSONDecodeError, TypeError): # Handle non-string/non-JSON output
        pass # Output was likely the "not found" message or an error string