from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage

# This is a helper to get the LLM instance into our tool functions
llm_instance = None
def set_llm_for_tools(llm):
    global llm_instance
    llm_instance = llm

# --- Tool Schemas ---

class GenerateItinerary(BaseModel):
    """
    Use this tool to generate a detailed, multi-day travel itinerary.
    Extract all details from the user's request.
    """
    destination: str = Field(description="The city or country the user wants to visit.")
    duration_days: int = Field(description="The total number of days for the trip.")
    budget_per_person: int = Field(description="The approximate budget per person in their local currency.")
    interests: List[str] = Field(description="A list of interests, like 'food', 'nightlife', 'adventure', 'culture'.")
    num_travelers: int = Field(description="The number of people going on the trip.")

class SuggestViralChallenge(BaseModel):
    """
    Use this tool to suggest a fun, short-form video challenge (like an Instagram Reel or YT Short)
    related to a travel destination.
    """
    destination: str = Field(description="The city or destination for which to create the challenge.")

class ToPlacePromoter(BaseModel):
    """Transfers work to a specialist who can provide exciting details about a single travel destination."""
    user_query: str = Field(description="The user's original question about a single place, like 'Tell me about Goa'.")

class ToPlaceComparator(BaseModel):
    """Transfers work to a specialist who can compare two or more travel destinations."""
    user_query: str = Field(description="The user's original question comparing places, like 'Goa or Shimla?'.")

class ToTransportation(BaseModel):
    """Transfers work to a specialist for questions about flights, trains, or travel routes."""
    user_query: str = Field(description="The user's original question about how to get somewhere.")

class ToAccommodation(BaseModel):
    """Transfers work to a specialist for questions about hotels, resorts, or places to stay."""
    user_query: str = Field(description="The user's original question about accommodation.")

class ToPacking(BaseModel):
    """Transfers work to a specialist who gives packing advice for a trip."""
    user_query: str = Field(description="The user's original question about what to pack.")

class CreatePoll(BaseModel):
    """Use this to create a poll in the group chat. Ask the user for the options if they are not provided."""
    question: str = Field(description="The main question for the poll.")
    options: List[str] = Field(description="A list of options for the poll, e.g., ['Mountains', 'Beaches'].")

class FlagAsIrrelevant(BaseModel):
    """
    Use this to flag a user's question as irrelevant or off-topic.
    This should be used for any question that is not about travel, including greetings.
    """
    reason: str = Field(description="A brief reason why the question is irrelevant, e.g., 'User said hi'.")

class CompleteOrEscalate(BaseModel):
    """
    Use this tool to signify you have answered the user's question.
    This returns control to the primary assistant.
    """
    reason: str = Field(description="A brief reason for finishing, e.g., 'Successfully provided details about Goa.'")


# --- Tool Implementations ---

def generate_itinerary_implementation(destination: str, duration_days: int, budget_per_person: int, interests: list, num_travelers: int):
    """Generates a travel itinerary by invoking the LLM with a specific prompt."""
    prompt = f"""You are a world-class travel agent. Create a detailed, day-by-day itinerary for a trip based on the following details.
    Present it in a visually appealing, easy-to-read markdown format. Include estimated costs where possible.

    - Destination: {destination}
    - Number of Travelers: {num_travelers}
    - Duration: {duration_days} days
    - Budget per Person: Approximately {budget_per_person}
    - Key Interests: {', '.join(interests)}

    Make the itinerary exciting and practical.
    """
    print("\n--- [TOOL CALLED: GenerateItinerary] ---")
    response = llm_instance.invoke([HumanMessage(content=prompt)])
    print("--- [ITINERARY GENERATED] ---\n")
    return response.content

def suggest_viral_challenge_implementation(destination: str):
    """Generates a viral challenge by invoking the LLM with a specific prompt."""
    prompt = f"""You are a viral marketing expert for a top social media brand.
    Create a fun, short, and engaging Instagram Reel or YouTube Short challenge called the '#{destination.replace(" ", "")}ReelChallenge'.
    The challenge should be themed around a trip to {destination}.
    List 3-5 specific, creative shots or actions the users should capture.
    Make it sound exciting and trendy.
    """
    print("\n--- [TOOL CALLED: SuggestViralChallenge] ---")
    response = llm_instance.invoke([HumanMessage(content=prompt)])
    print("--- [CHALLENGE GENERATED] ---\n")
    return response.content

def create_poll_in_group(question: str, options: list):
    """Creates a poll in the group chat."""
    print("\n--- [TOOL CALLED: CreatePoll] ---")
    print(f"Poll Question: {question}")
    print(f"Poll Options: {options}")
    print("---------------------------------\n")
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