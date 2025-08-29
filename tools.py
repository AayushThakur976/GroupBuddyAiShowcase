from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage

llm_instance = None
def set_llm_for_tools(llm):
    global llm_instance
    llm_instance = llm

# --- Tool Schemas ---
class GenerateItinerary(BaseModel):
    """
    Use this to generate a travel itinerary. This single tool can handle both specific destinations
    (e.g., "Goa") and vague descriptions (e.g., "a cloudy place").
    """
    destination_or_description: Optional[str] = Field(default=None, description="The user's core request, cleaned up. E.g., 'a cloudy place for the weekend' or 'Goa'.")
    duration_days: Optional[int] = Field(default=None, description="The total number of days for the trip.")
    budget_per_person: Optional[int] = Field(default=None, description="The approximate budget per person.")
    num_travelers: Optional[int] = Field(default=None, description="The number of people going on the trip.")
    interests: Optional[List[str]] = Field(default=None, description="An optional list of interests.")

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
def generate_itinerary_implementation(destination_or_description: Optional[str] = None, duration_days: Optional[int] = None, budget_per_person: Optional[int] = None, num_travelers: Optional[int] = None, interests: Optional[list] = None):
    """Generates a travel itinerary, making creative assumptions for any missing details."""
    # THIS PROMPT HAS BEEN HARDENED TO PREVENT NON-INDIAN RESPONSES
    prompt = f"""You are a world-class travel agent **based in Delhi, India, specializing in trips originating from North India.** A user, who is also in India, wants a trip plan.

    **CRITICAL CONTEXT: ALL suggestions MUST be in INDIA and easily accessible from Delhi for a weekend trip.** Do not suggest international locations under any circumstances.

    Their core request is: **'{destination_or_description or "a popular weekend getaway from Delhi"}'**.

    Your task is to interpret this request with the India-only context in mind.
    - If it's a specific Indian place, plan for it.
    - If it's a description (like 'a cloudy place'), you MUST CHOOSE the single best-fit destination **near Delhi (e.g., in Himachal Pradesh, Uttarakhand)** and then create the full day-by-day plan for that chosen place. You MUST NOT just list options; you must create the complete itinerary.

    Create a complete, day-by-day itinerary using the following details. Make plausible assumptions for any missing information.
    - Number of Travelers: {num_travelers or 2}
    - Duration: {duration_days or 2} days
    - Budget per Person: Approximately INR {budget_per_person or 2500}
    - Key Interests: {', '.join(interests) if interests else "a general mix of popular activities"}

    Present the chosen destination and the full itinerary in a beautiful markdown format.
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