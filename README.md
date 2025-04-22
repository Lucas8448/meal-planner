# Multi-Agent Meal Planner API

This project implements a multi-agent system using LangGraph and FastAPI to generate weekly meal plans and shopping lists based on grocery store deals in Norway.

## System Overview

The system utilizes four distinct agents working sequentially:

1.  **Agent 1: Deal Hunter**

    - **Goal:** Find Norwegian grocery products with recent price drops suitable for common dinners.
    - **Process:**
      - Generates a list of common Norwegian dinner ingredient search terms based on an initial query.
      - Uses the `search_products` tool (powered by the Kassal.app API) to find products matching these terms that have a confirmed price drop.
    - **Output:** A list of search terms used and a list of `DealInfo` objects for products found with price drops.

2.  **Agent 2: Meal Strategist**

    - **Goal:** Choose a primary store based on deals, create a 7-day dinner plan, and identify missing essential ingredients.
    - **Process:**
      - Analyzes deals found by Agent 1, grouped by store.
      - Selects a single store with a good cluster of deals.
      - Creates a 7-day dinner plan (for 2 people) using deals _only_ from the chosen store and considering ingredients the user already has (`on_hand_ingredients`).
      - Identifies essential generic ingredients needed for the plan that are _not_ in the deals or the user's on-hand list.
    - **Output:** The chosen store name, a list of `MealPlanItem` objects, and a list of missing ingredient names.

3.  **Agent 3: Bargain Scout**

    - **Goal:** Find one standard, reasonably priced option for each missing ingredient identified by Agent 2, searching across all nearby stores.
    - **Process:**
      - For each missing ingredient name:
        - Uses `search_products` (with price drop filter _disabled_) to find potential matches.
        - Uses `get_product_details` to get more info on promising candidates.
        - Selects the single best standard option based on unit type and price.
    - **Output:** A list of dictionaries (`cheapest_ingredients_info`), each describing the best option found for a missing ingredient.

4.  **Agent 4: List Consolidator**
    - **Goal:** Create a final shopping list grouped by store, combining meal plan deals and found missing ingredients.
    - **Process:**
      - Extracts all deal items used in the meal plan (from Agent 2).
      - Collects the staple items found by Agent 3.
      - Combines these lists and organizes them by store.
    - **Output:** A dictionary where keys are store names and values are lists of `ShoppingListItem` objects to buy.

## Setup

1.  **Clone the Repository:**

    ```bash
    git clone <your-repo-url>
    cd meal-planner
    ```

2.  **Create Environment File:**
    Create a file named `.env` in the project root directory and add the following variables:

    ```dotenv
    # Get from OpenAI Platform
    OPENAI_API_KEY="sk-..."

    # Get from Kassal.app API access
    kassalapp_api_key="..."

    # Your location for nearby store filtering
    location_latitude="YOUR_LATITUDE"    # e.g., 59.9139
    location_longitude="YOUR_LONGITUDE" # e.g., 10.7522
    location_radius="5"                 # Search radius in kilometers
    ```

3.  **Install Dependencies:**
    You'll need Python 3 installed. Install the required packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Running the API

Start the FastAPI server using Uvicorn:

```bash
python3 main.py
```

The server will start, typically on `http://0.0.0.0:8000`. You'll see output like:

```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
## Starting Meal Planner API Server...
```

The server supports hot-reloading, so changes to `main.py` should restart it automatically during development.

## API Usage

The API exposes one main endpoint:

- **Endpoint:** `/plan-meals`
- **Method:** `POST`
- **Request Body:** JSON object containing:
  - `query` (string, optional): The initial query for the meal planner (defaults to a generic query).
  - `on_hand_ingredients` (list of strings, optional): A list of ingredient names the user already possesses.
- **Response:** A JSON object representing the final `MealPlannerState`, including the generated `meal_plan`, `shopping_list`, and intermediate agent outputs.

**Example using `curl`:**

```bash
curl -X POST http://localhost:8000/plan-meals \
-H "Content-Type: application/json" \
-d '{
  "query": "Plan a week of fish and chicken dinners",
  "on_hand_ingredients": ["salt", "pepper", "olive oil", "rice", "pasta"]
}'
```

**Example Response Snippet:**

```json
{
  "initial_query": "Plan a week of fish and chicken dinners",
  "on_hand_ingredients": ["salt", "pepper", "olive oil", "rice", "pasta"],
  "search_terms": ["torsk", "laks", "kyllingfilet", ...],
  "found_deals": [...],
  "chosen_store": "SPAR",
  "meal_plan": [
    {
      "meal_name": "Day 1: Torskefilet with Rice",
      "deals_used": [{"id": 10135, "name": "Torskefilet Msc 800g First Price", ..., "image_url": "..."}],
      "on_hand_used": ["rice", "salt", "pepper", "olive oil"]
    },
    ...
  ],
  "missing_ingredients": ["butter", "lemon"],
  "cheapest_ingredients_info": [...],
  "shopping_list": {
    "SPAR": [
      {"name": "Torskefilet Msc 800g First Price", "price": 97.9, ..., "image_url": "..."}
      ...
    ],
    "KIWI": [
       {"name": "Sitroner pk", "price": 25.0, ..., "image_url": "..."}
    ]
  },
  "agent_outcome": { ... } // Contains the final shopping list again
}
```
