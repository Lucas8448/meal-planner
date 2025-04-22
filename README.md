# Meal Planner

A service that creates weekly meal plans based on current grocery deals and ingredients on hand.

## Features

- Finds discounted grocery products in Norway using the Kassal.app API
- Creates a 7-day dinner plan for two people using deals from a single store
- Considers ingredients you already have on hand
- Generates a shopping list with all items needed from a single store
- Identifies missing essential ingredients

## API Usage

### Endpoint

```
POST /api/plan-meals
```

### Request

```json
{
  "query": "Find good dinner deals in Norway for two people",
  "on_hand_ingredients": [
    "pasta",
    "rice",
    "onions",
    "garlic",
    "olive oil",
    "salt",
    "pepper",
    "potatoes"
  ]
}
```

### Response

The API returns a JSON response with the following structure:

```json
{
  "chosen_store": "SPAR",
  "meal_plan": [
    {
      "meal_name": "Day 1: Pan-fried Torskefilet with Boiled Potatoes",
      "deals_used": [
        {
          "id": 10135,
          "name": "Torskefilet Msc 800g First Price",
          "current_price": 97.9,
          "previous_price": 99.9,
          "price_drop_percentage": 2.0,
          "currency": "NOK",
          "store": "SPAR",
          "image_url": "https://bilder.ngdata.no/7035620084003/kmh/large.jpg"
        }
      ],
      "on_hand_used": ["potatoes", "olive oil", "salt", "pepper"],
      "notes": "Likely leftovers"
    }
  ],
  "missing_ingredients": ["butter", "milk", "cream"],
  "shopping_list": {
    "SPAR": [
      {
        "name": "Torskefilet Msc 800g First Price",
        "price": 97.9,
        "currency": "NOK",
        "notes": "Deal item",
        "image_url": "https://bilder.ngdata.no/7035620084003/kmh/large.jpg"
      }
    ]
  }
}
```

### Response Time

The API typically takes approximately 2-3 minutes to respond due to the following operations:

- Searching for current deals across multiple product categories
- Analyzing deals to create an optimal 7-day meal plan
- Searching for pricing on missing ingredients
- Consolidating results into a shopping list

## Setup

1. Clone the repository
2. Create a `.env` file with required credentials:

```
kassalapp_api_key=your_key_here
OPENAI_API_KEY=your_key_here
location_latitude=your_latitude
location_longitude=your_longitude
location_radius=4000
```

3. Install dependencies: `pip install -r requirements.txt`
4. Run the server: `python main.py`

## Architecture

The meal planner uses a multi-agent workflow with the following steps:

1. Deal Hunter: Searches for products with price drops
2. Meal Strategist: Creates a meal plan using deals from a single store
3. Bargain Scout: Finds optimal prices for missing ingredients
4. List Consolidator: Creates a shopping list from a single store

## Project Structure

The project is organized using a modular package structure:

```
meal_planner/
├── __init__.py
├── agents/                # AI agent implementations
│   ├── __init__.py
│   ├── agents_registry.py # Registry for accessing agent instances
│   ├── bargain_scout.py   # Agent that finds best prices for missing ingredients
│   ├── deal_hunter.py     # Agent that finds grocery deals
│   ├── list_consolidator.py # Agent that creates the final shopping list
│   └── meal_strategist.py # Agent that creates the meal plan
├── api/                   # FastAPI implementation
│   ├── __init__.py
│   ├── app.py             # FastAPI app configuration
│   ├── models.py          # API request/response models
│   └── routes.py          # API route definitions
├── config/                # Application configuration
│   ├── __init__.py
│   └── settings.py        # Settings from environment variables
├── models/                # Data models
│   ├── __init__.py
│   └── state.py           # State models for the workflow
├── tools/                 # External API tools
│   ├── __init__.py
│   └── kassalapp.py       # Tools for the Kassalapp API
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── json_helpers.py    # JSON processing utilities
└── workflow.py            # LangGraph workflow definition
```

## Technologies Used

- **FastAPI**: Web framework for building APIs
- **LangGraph**: For building multi-agent workflows
- **LangChain**: For building LLM applications
- **OpenAI**: For language model capabilities
- **Kassalapp API**: For accessing Norwegian grocery deals

## License

This project is licensed under the MIT License.
