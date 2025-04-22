"""Tools for interacting with the Kassalapp API."""

import json
import requests
from typing import Dict, List, Optional, Set, Any, Union
from pydantic import BaseModel, Field

from langchain.tools import tool

from meal_planner.models.state import DealInfo
from meal_planner.config.settings import settings, _nearby_store_groups_cache


class NearbyStoresInput(BaseModel):
    """Input model for the get_nearby_stores tool."""
    pass


@tool("get_nearby_stores", args_schema=NearbyStoresInput)
def get_nearby_stores() -> str:
    """(Internal) Fetch nearby stores based on env vars as JSON string."""
    lat, lng, km = settings.location_latitude, settings.location_longitude, settings.location_radius
    if not all([lat, lng, km]):
        return "Error: Missing location environment variables."
    
    url = f"{settings.kassalapp_base_url}/physical-stores?size=100&lat={lat}&lng={lng}&km={km}"
    headers = {"Authorization": f"Bearer {settings.kassalapp_api_key}"}
    
    print(f"DEBUG (get_nearby_stores): Requesting URL: {url}")
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        print(f"DEBUG (get_nearby_stores): Request failed: {e}")
        return f"Error fetching nearby stores: {e}"


def get_cached_nearby_store_groups() -> Set[str]:
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

    _nearby_store_groups_cache = nearby_store_groups  # Cache result (even empty set)
    return nearby_store_groups


class SearchProductsInput(BaseModel):
    """Input model for the search_products tool."""
    search: str = Field(description="The Norwegian search term for products.")
    filter_by_price_drop: bool = Field(
        default=True, 
        description="Whether to filter results to only include products with a recent price drop."
    )


@tool("search_products", args_schema=SearchProductsInput)
def search_products(search: str, filter_by_price_drop: bool = True) -> List[DealInfo] | str:
    """Search products by term, optionally filter by nearby stores (cached) & price drops.
    
    If filter_by_price_drop is True (default), returns simplified list of products with actual price drops:
    [{'id', 'name', 'current_price', 'previous_price', 'price_drop_percentage', 'currency', 'store'}]
    
    If filter_by_price_drop is False, returns simplified list of found products:
    [{'id', 'name', 'current_price', 'currency', 'store'}]
    """
    if not isinstance(search, str) or not search.strip():
        return "Error: search term must be a non-empty string."

    nearby_store_groups = get_cached_nearby_store_groups()
    url = f"{settings.kassalapp_base_url}/products?search={search}&size=100"
    headers = {"Authorization": f"Bearer {settings.kassalapp_api_key}"}
    
    print(f"DEBUG (search_products): Requesting URL: {url} | Nearby Groups: {nearby_store_groups or 'None/Empty'} | Filter Drops: {filter_by_price_drop}")

    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        products_raw = data.get("data")

        if not isinstance(products_raw, list):
            return "Error: Unexpected API response format (missing 'data' list)."

        print(f"DEBUG (search_products): Received {len(products_raw)} raw products for '{search}'.")

        final_products = []
        for prod in products_raw:
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

            # Process based on filter setting
            if filter_by_price_drop:
                # Process price history
                raw_history = prod.get("price_history", [])
                processed_history = []
                if raw_history:
                    try:
                        sorted_history = sorted(raw_history, key=lambda x: x.get('date', ''), reverse=True)
                        for entry in sorted_history[:10]:
                            price_str = entry.get('price')
                            if price_str is not None:
                                try: 
                                    processed_history.append(float(price_str))
                                except (ValueError, TypeError): 
                                    pass
                    except Exception as e:
                        print(f"DEBUG (search_products): Error processing history for prod {prod_id}: {e}")
                        processed_history = []

                if not processed_history:  # Need history to check for drops
                    continue

                # Find previous price
                previous_price = None
                for price in processed_history:
                    if price != current_price:
                        previous_price = price
                        break

                # Price Drop Check & Calculation
                if previous_price is not None and previous_price > current_price:
                    try:
                        price_drop_percentage = round(((previous_price - current_price) / previous_price) * 100, 2)
                        if price_drop_percentage > 0:  # Double check it's a drop
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
                        continue  # Skip if previous price was 0

            else:  # filter_by_price_drop is False - just add basic details
                final_products.append({
                    "id": prod_id,
                    "name": prod_name,
                    "current_price": current_price,
                    "currency": "NOK",
                    "store": store_name,
                    "image_url": prod.get("image"),
                })

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


class ProductDetailsInput(BaseModel):
    """Input model for the get_product_details tool."""
    product_id: int | str = Field(description="The EAN or internal ID of the product.")


@tool("get_product_details", args_schema=ProductDetailsInput)
def get_product_details(product_id: int | str) -> Dict[str, Any] | str:
    """Fetch detailed information for a single product by its EAN or ID."""
    if not product_id:
        return "Error: product_id must be provided."

    url = f"{settings.kassalapp_base_url}/products/{product_id}"
    headers = {"Authorization": f"Bearer {settings.kassalapp_api_key}"}
    
    print(f"DEBUG (get_product_details): Requesting URL: {url}")

    try:
        resp = requests.get(url, headers=headers)
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
        return {k: v for k, v in relevant_details.items() if v is not None}  # Clean nulls

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