"""Agent responsible for searching products with price drops."""

from textwrap import dedent
from typing import Dict, Any, List

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from meal_planner.config.settings import settings
from meal_planner.tools.kassalapp import search_products
from meal_planner.models.state import MealPlannerState, DealInfo
from meal_planner.utils.json_helpers import parse_llm_json_output


class ProductSearchAgent:
    """Agent that searches for grocery products with price drops based on user query."""
    
    def __init__(self):
        """Initialize the Product Search agent."""
        self.prompt = ChatPromptTemplate.from_messages([
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
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.tools = [search_products]
        self.llm = ChatOpenAI(
            model=settings.llm_model, 
            temperature=settings.llm_temperature
        )
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=False, 
            max_iterations=25
        )
    
    def run(self, state: MealPlannerState) -> MealPlannerState:
        """Run the Product Search agent to find products with price drops.
        
        Args:
            state: The current state of the meal planning workflow.
            
        Returns:
            Updated state with search terms and found deals.
        """
        print("--- Running Agent 1: Product Search ---")
        initial_query = state['initial_query']
        
        # Invoke the agent
        result = self.executor.invoke({"input": initial_query})
        agent_output = result.get("output", "{}")
        
        print(f"DEBUG (Agent 1 Output): {agent_output}")
        
        # Parse and validate the output
        parsed_output, error = parse_llm_json_output(agent_output)
        
        if error:
            updated_state = state.copy()
            updated_state["search_terms"] = []
            updated_state["found_deals"] = []
            updated_state["agent_outcome"] = {"error": error}
            return updated_state
        
        try:
            search_terms = parsed_output.get("search_terms", [])
            found_deals = parsed_output.get("found_deals", [])
            
            if not isinstance(search_terms, list) or not isinstance(found_deals, list):
                raise ValueError("Invalid data types in parsed output")
                
            print(f"DEBUG (Agent 1 Parsed): Found {len(found_deals)} deals using {len(search_terms)} terms.")
            
            # Update and return state
            updated_state = state.copy()
            updated_state["search_terms"] = search_terms
            updated_state["found_deals"] = found_deals
            updated_state["agent_outcome"] = parsed_output
            return updated_state
            
        except Exception as e:
            print(f"ERROR (Agent 1): {str(e)}")
            updated_state = state.copy()
            updated_state["search_terms"] = []
            updated_state["found_deals"] = []
            updated_state["agent_outcome"] = {"error": f"Agent 1 Error: {str(e)}"}
            return updated_state 