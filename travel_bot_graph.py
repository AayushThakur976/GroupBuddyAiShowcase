import os
from typing import Annotated, List, Literal, Optional, Union, Any
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import Runnable, RunnableConfig

from tools import (
    GenerateItinerary,
    ToPlacePromoter, ToPlaceComparator, ToTransportation, ToAccommodation, ToPacking, CompleteOrEscalate,
    CreatePoll, FlagAsIrrelevant, SuggestViralChallenge,
    generate_itinerary_implementation,
    create_poll_in_group, to_place_promoter, to_place_comparator, to_transportation,
    to_accommodation, to_packing, flag_as_irrelevant,
    suggest_viral_challenge_implementation,
    set_llm_for_tools
)
from prompts import (
    PRIMARY_ASSISTANT_PROMPT, PLAN_PRESENTER_PROMPT,
    FINAL_RESPONSE_PROMPT,
    PLACE_PROMOTER_PROMPT, PLACE_COMPARATOR_PROMPT,
    TRANSPORTATION_PROMPT, ACCOMMODATION_PROMPT, PACKING_PROMPT
)

# --- LLM Configuration ---
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=f"https://{os.getenv('AZURE_INSTANCE_NAME')}.openai.azure.com/",
    deployment_name=os.getenv("AZURE_DEPLOYMENT"),
    api_version=os.getenv("AZURE_API_VERSION"),
    temperature=0.7,
)

set_llm_for_tools(llm)

# --- State Definition ---
def update_dialog_stack(left: List[str], right: Optional[str]) -> List[str]:
    if right is None: return left
    if right == "pop": return left[:-1] if left else []
    return left + [right]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    dialog_state: Annotated[
        List[Literal["place_promoter", "place_comparator", "transportation", "accommodation", "packing"]],
        update_dialog_stack,
    ]

# --- Assistant Node Wrapper ---
class AssistantNode:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable
    def __call__(self, state: State, config: RunnableConfig):
        result = self.runnable.invoke(state)
        return {"messages": [result]}

# --- Agent and Tool Definitions ---
final_responder = AssistantNode(FINAL_RESPONSE_PROMPT | llm)

primary_assistant_tools_list = [
    GenerateItinerary,
    SuggestViralChallenge,
    ToPlacePromoter, ToPlaceComparator, ToTransportation, ToAccommodation, ToPacking,
    CreatePoll, FlagAsIrrelevant
]
primary_assistant = AssistantNode(PRIMARY_ASSISTANT_PROMPT | llm.bind_tools(primary_assistant_tools_list))

primary_tool_node = ToolNode([
    generate_itinerary_implementation,
    suggest_viral_challenge_implementation,
    create_poll_in_group,
    to_place_promoter,
    to_place_comparator,
    to_transportation,
    to_accommodation,
    to_packing,
    flag_as_irrelevant
])

plan_presenter_assistant = AssistantNode(PLAN_PRESENTER_PROMPT | llm)

chitchat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are TravelBuddy, a friendly AI travel planner. The user has asked a question that is not related to travel, or has just said hello. Your job is to respond gracefully and pivot the conversation back to travel."""),
    MessagesPlaceholder(variable_name="messages")
])
chitchat_assistant = AssistantNode(chitchat_prompt_template | llm)

specialist_assistant_tools = [CompleteOrEscalate]
specialist_tool_node = ToolNode(specialist_assistant_tools)

place_promoter_assistant = AssistantNode(PLACE_PROMOTER_PROMPT | llm.bind_tools(specialist_assistant_tools))
place_comparator_assistant = AssistantNode(PLACE_COMPARATOR_PROMPT | llm.bind_tools(specialist_assistant_tools))
transportation_assistant = AssistantNode(TRANSPORTATION_PROMPT | llm.bind_tools(specialist_assistant_tools))
accommodation_assistant = AssistantNode(ACCOMMODATION_PROMPT | llm.bind_tools(specialist_assistant_tools))
packing_assistant = AssistantNode(PACKING_PROMPT | llm.bind_tools(specialist_assistant_tools))

# --- Graph Routing Functions ---
def route_to_specialist(state: State) -> Union[str, Any]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, ToolMessage):
        return END
    if last_message.name == GenerateItinerary.__name__:
        return "plan_presenter"
    if last_message.name == SuggestViralChallenge.__name__:
        return "primary_assistant"
    if last_message.name == CreatePoll.__name__:
        return "final_responder"
    if last_message.name == FlagAsIrrelevant.__name__:
        return "chitchat_assistant"
    if last_message.name == ToPlacePromoter.__name__: return "enter_place_promoter"
    if last_message.name == ToPlaceComparator.__name__: return "enter_place_comparator"
    if last_message.name == ToTransportation.__name__: return "enter_transportation"
    if last_message.name == ToAccommodation.__name__: return "enter_accommodation"
    if last_message.name == ToPacking.__name__: return "enter_packing"
    if last_message.name == CompleteOrEscalate.__name__:
        return "pop_dialog_state"
    return "primary_assistant"

def route_from_primary(state: State) -> Union[Literal["primary_tools"], Any]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "primary_tools"
    return END

def get_active_specialist(state: State) -> str:
    if active_specialist := state.get("dialog_state"):
        return active_specialist[-1]
    return "primary_assistant"

def route_specialist_assistant(state: State) -> Union[Literal["specialist_tools"], Any]:
    if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls:
        return "specialist_tools"
    return END

# --- Graph Builder ---
builder = StateGraph(State)
builder.add_node("primary_assistant", primary_assistant)
builder.add_node("primary_tools", primary_tool_node)
builder.add_node("plan_presenter", plan_presenter_assistant)
builder.add_node("specialist_tools", specialist_tool_node)
builder.add_node("final_responder", final_responder)
builder.add_node("chitchat_assistant", chitchat_assistant)
builder.add_node("place_promoter", place_promoter_assistant)
builder.add_node("place_comparator", place_comparator_assistant)
builder.add_node("transportation", transportation_assistant)
builder.add_node("accommodation", accommodation_assistant)
builder.add_node("packing", packing_assistant)
builder.add_node("enter_place_promoter", lambda s: {"dialog_state": "place_promoter"})
builder.add_node("enter_place_comparator", lambda s: {"dialog_state": "place_comparator"})
builder.add_node("enter_transportation", lambda s: {"dialog_state": "transportation"})
builder.add_node("enter_accommodation", lambda s: {"dialog_state": "accommodation"})
builder.add_node("enter_packing", lambda s: {"dialog_state": "packing"})
builder.add_node("pop_dialog_state", lambda s: {"dialog_state": "pop"})

# --- Graph Flow ---
builder.set_conditional_entry_point(get_active_specialist, {
    "primary_assistant": "primary_assistant", "place_promoter": "place_promoter",
    "place_comparator": "place_comparator", "transportation": "transportation",
    "accommodation": "accommodation", "packing": "packing",
})
builder.add_conditional_edges("primary_assistant", route_from_primary, {"primary_tools": "primary_tools", END: END})
builder.add_conditional_edges("primary_tools", route_to_specialist, {
    "plan_presenter": "plan_presenter", "final_responder": "final_responder",
    "chitchat_assistant": "chitchat_assistant", "enter_place_promoter": "enter_place_promoter",
    "enter_place_comparator": "enter_place_comparator", "enter_transportation": "enter_transportation",
    "enter_accommodation": "enter_accommodation", "enter_packing": "enter_packing",
    "primary_assistant": "primary_assistant", END: END
})
builder.add_edge("plan_presenter", END)
builder.add_edge("final_responder", END)
builder.add_edge("chitchat_assistant", END)
for specialist in ["place_promoter", "place_comparator", "transportation", "accommodation", "packing"]:
    builder.add_edge(f"enter_{specialist}", specialist)
    builder.add_conditional_edges(specialist, route_specialist_assistant, {"specialist_tools": "specialist_tools", END: END})
builder.add_edge("specialist_tools", "pop_dialog_state")
builder.add_edge("pop_dialog_state", "primary_assistant")

# --- Compilation ---
memory = MemorySaver()
travel_agent_graph = builder.compile(checkpointer=memory)

print("✅ Final, robust Travel Agent graph with Presenter node compiled!")