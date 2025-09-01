from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage
from datetime import datetime, timedelta

llm_instance = None
def set_llm_for_tools(llm):
    """Allows the main graph builder to pass the LLM instance to this file."""
    global llm_instance
    llm_instance = llm

# --- Tool Schemas ---

class GenerateItinerary(BaseModel):
    """
    Use this to generate a travel itinerary. This tool requires a destination and ideally a starting location (origin).
    """
    destination_or_description: Optional[str] = Field(default=None, description="The user's core request, cleaned up. E.g., 'a cloudy place for the weekend' or 'Goa'.")
    origin: Optional[str] = Field(default=None, description="The user's starting location for the trip, if mentioned.")
    duration_days: Optional[int] = Field(default=None, description="The total number of days for the trip.")
    budget_per_person: Optional[int] = Field(default=None, description="The approximate budget per person.")
    num_travelers: Optional[int] = Field(default=None, description="The number of people going on the trip.")
    interests: Optional[List[str]] = Field(default=None, description="An optional list of interests.")

class RequestOriginLocation(BaseModel):
    """Use this tool when a user wants to plan a trip but has not specified their starting location (origin)."""
    # FIXED: Added a dummy argument to ensure content is always returned.
    question: str = Field(description="The question to ask the user. Should always be 'Where are you traveling from?'")

class GetTravelLinks(BaseModel):
    """Gets pre-filled travel links for buses (Redbus) and trains (IRCTC)."""
    origin: str = Field(description="The starting city, e.g., 'Delhi'")
    destination: str = Field(description="The destination city, e.g., 'Manali'")

class SuggestViralChallenge(BaseModel):
    """Suggests a fun, short-form video challenge for a travel destination."""
    destination: str = Field(description="The city or destination for which to create the challenge.")

class ToPlacePromoter(BaseModel):
    """Transfers work to a specialist for details about a single travel destination."""
    user_query: str = Field(description="The user's original question about a single place.")

class ToPlaceComparator(BaseModel):
    """Transfers work to a specialist who can compare two or more travel destinations."""
    user_query: str = Field(description="The user's original question comparing places.")

class ToTransportation(BaseModel):
    """Transfers work to a specialist for questions about flights, trains, or travel routes."""
    user_query: str = Field(description="The user's original question about how to get somewhere.")

class ToAccommodation(BaseModel):
    """Transfers work to a specialist for questions about hotels or places to stay."""
    user_query: str = Field(description="The user's original question about accommodation.")

class ToPacking(BaseModel):
    """Transfers work to a specialist who gives packing advice for a trip."""
    user_query: str = Field(description="The user's original question about what to pack.")

class ToWeather(BaseModel):
    """Transfers work to a specialist for questions about the weather for a trip."""
    user_query: str = Field(description="The user's original question about the weather.")

class CreatePoll(BaseModel):
    """Use this to create a poll in the group chat."""
    question: str = Field(description="The main question for the poll.")
    options: List[str] = Field(description="A list of options for the poll.")

class FlagAsIrrelevant(BaseModel):
    """Flags a user's question as irrelevant or off-topic."""
    reason: str = Field(description="A brief reason why the question is irrelevant.")

class CompleteOrEscalate(BaseModel):
    """Signifies you have answered the user's question and returns control to the primary assistant."""
    reason: str = Field(description="A brief reason for finishing.")


# --- Tool Implementations ---

# FIXED: Modified the function to accept the dummy argument.
def RequestOriginLocation(question: str) -> str:
    """Returns a polite question asking the user for their starting location."""
    return "Sounds like a fun trip! To give you the best suggestions, where will you be traveling from?"

def get_travel_links(origin: str, destination: str) -> str:
    """Builds and returns formatted travel links."""
    origin_formatted = origin.title()
    destination_formatted = destination.title()
    travel_date = (datetime.now() + timedelta(days=1)).strftime("%d-%b-%Y")
    
    # --- MODIFIED LINE ---
    # Changed from a pre-filled f-string to a plain URL.
    redbus_url = "https://www.redbus.in/"
    
    irctc_url = "https://www.irctc.co.in/nget/train-search"
    
    # --- MODIFIED RETURN TEXT ---
    # Changed "pre-filled links" to "helpful links".
    return (
        f"Here are some helpful links for your trip from {origin} to {destination}:\n\n"
        f"🚌 For Buses: [Search for buses on Redbus]({redbus_url})\n\n"
        f"🚆 For Trains: [Search for trains on the IRCTC Website]({irctc_url})"
    )
def generate_itinerary_implementation(destination_or_description: Optional[str] = None, origin: Optional[str] = None, duration_days: Optional[int] = None, budget_per_person: Optional[int] = None, num_travelers: Optional[int] = None, interests: Optional[list] = None):
    """Generates a travel itinerary, making creative assumptions for any missing details."""
    start_location = origin or "Delhi"
    prompt = f"""You are a world-class travel agent. A user wants a trip plan starting from '{start_location}'.

    **CRITICAL CONTEXT: ALL suggestions MUST be in INDIA.** Do not suggest international locations.

    Their core request is: **'{destination_or_description or f"a popular weekend getaway from {start_location}"}'**.

    Your task is to interpret this request.
    - If it's a specific Indian place, plan for it.
    - If it's a description (like 'a cloudy place'), you MUST CHOOSE the single best-fit destination easily accessible from '{start_location}' and then create the full day-by-day plan for that chosen place.

    Create a complete, day-by-day itinerary using the following details. Make plausible assumptions for any missing information.
    - Number of Travelers: {num_travelers or 2}
    - Duration: {duration_days or 2} days
    - Budget per Person: Approximately INR {budget_per_person or 2500}
    - Key Interests: {', '.join(interests) if interests else "a general mix of popular activities"}

    ## IMPORTANT OUTPUT INSTRUCTION
    Present the itinerary in a **conversational, paragraph-based style**. Describe the plan day-by-day in a natural way, as if you were explaining it to a friend. **DO NOT use markdown headers (`###`), bolding (`**`), or itemized lists (`-`)**.
    """
    response = llm_instance.invoke([HumanMessage(content=prompt)])
    return response.content

def suggest_viral_challenge_implementation(destination: str):
    """Generates a viral challenge by invoking the LLM with a specific prompt."""
    prompt = f"""You are a viral marketing expert. Create a fun, short, and engaging Instagram Reel challenge for a trip to {destination}. List 3-5 creative shots or actions the users should capture."""
    response = llm_instance.invoke([HumanMessage(content=prompt)])
    return response.content

def create_poll_in_group(question: str, options: list):
    """Creates a poll in the group chat."""
    print(f"--- POLL CREATED ---\nQuestion: {question}\nOptions: {options}\n--------------------")
    return "Poll created successfully!"

def flag_as_irrelevant(reason: str) -> str:
    """Flags a message as irrelevant to travel planning."""
    return "Acknowledged as non-travel."

def to_place_promoter(user_query: str) -> str:
    """Delegates the conversation to the place promoter specialist."""
    return f"Delegating to place promoter with query: {user_query}"

def to_place_comparator(user_query: str) -> str:
    """Delegates the conversation to the place comparator specialist."""
    return f"Delegating to place comparator with query: {user_query}"

def to_transportation(user_query: str) -> str:
    """Delegates the conversation to the transportation specialist."""
    return f"Delegating to transportation specialist with query: {user_query}"

def to_accommodation(user_query: str) -> str:
    """Delegates the conversation to the accommodation specialist."""
    return f"Delegating to accommodation specialist with query: {user_query}"

def to_packing(user_query: str) -> str:
    """Delegates the conversation to the packing specialist."""
    return f"Delegating to packing specialist with query: {user_query}"

def to_weather(user_query: str) -> str:
    """Delegates the conversation to the weather specialist."""
    return f"Delegating to weather specialist with query: {user_query}"