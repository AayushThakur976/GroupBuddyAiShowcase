import os
import uuid
import traceback
import requests
import threading
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from langchain_core.messages import HumanMessage, AIMessage

# --- Load Environment Variables ---
load_dotenv()

# --- Custom Travel Bot Import ---
from travel_bot_graph import travel_agent_graph, llm

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Proactive Messaging Functions ---
def trigger_proactive_suggestion():
    with app.app_context():
        print(f"[{datetime.now()}] 🚀 Running proactive suggestion job...")
        today = datetime.now().strftime("%A, %B %d, %Y")
        prompt = f"""You are a fun, friendly travel buddy in a group chat for people based in Noida, India.
        Today is {today}.
        Your task is to proactively suggest one exciting and feasible weekend trip from Noida.
        Keep it short and exciting, and end with an engaging question.
        Use a fun, enthusiastic tone and include emojis.
        """
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            suggestion = response.content
            print(f"💡 Generated Suggestion: {suggestion}")
            post_message_to_group(suggestion)
        except Exception as e:
            print(f"Error during proactive suggestion: {e}")

def post_message_to_group(message: str):
    webhook_url = os.getenv("GROUP_CHAT_WEBHOOK_URL")
    if not webhook_url or webhook_url == "your_webhook_url_here":
        print("⚠️ Webhook URL not configured. Cannot post message.")
        return
    try:
        payload = {"text": message}
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        print(f"✅ Message posted successfully to the group: '{message[:50]}...'")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to post message to group: {e}")

# --- Background Processing ---
def process_in_background(user_message, session_id):
    with app.app_context():
        print(f"🤖 Starting background processing for session: {session_id}")
        try:
            post_message_to_group("Thinking about that for you, buddy...")

            config = {"configurable": {"thread_id": session_id}}
            graph_input = {"messages": [HumanMessage(content=user_message)]}
            final_state = travel_agent_graph.invoke(graph_input, config=config)

            agent_response = "I'm not sure how to respond to that, buddy."
            if messages := final_state.get("messages"):
                for message in reversed(messages):
                    if isinstance(message, AIMessage) and not message.tool_calls:
                        agent_response = message.content
                        break
            
            post_message_to_group(agent_response)

        except Exception as e:
            print(f"Error in background thread: {e}")
            traceback.print_exc()
            post_message_to_group("Sorry, I ran into a problem while thinking about that.")

# --- API Endpoint ---
@app.route("/api/travel/chat", methods=["POST"])
def chat():
    """
    Handles chat messages with the corrected deep parser for Google Chat events.
    """
    try:
        event_data = request.get_json()

        # --- CORRECTED DEEP PARSING LOGIC ---
        user_message = ""
        session_id = ""

        # First, try to parse the complex Google Chat event structure
        if "chat" in event_data and "messagePayload" in event_data["chat"]:
            message_payload = event_data["chat"]["messagePayload"]
            message_field = message_payload.get("message", {})
            
            if isinstance(message_field, dict):
                # For mentions, 'argumentText' is the clean message. Fallback to 'text'.
                user_message = message_field.get("argumentText", "").strip() or message_field.get("text", "").strip()

            # The session ID is the name of the space
            session_id = message_payload.get("space", {}).get("name", "")
        
        # If that fails, fall back to the simple format for Postman tests
        else:
            message_field = event_data.get("message", "")
            if isinstance(message_field, str):
                user_message = message_field.strip()
            session_id = event_data.get("session_id", "")
        # ------------------------------------

        if not user_message:
            return jsonify({}) # Ignore events without text (like adding the bot)

        if not session_id:
            session_id = f"test-session-{uuid.uuid4()}"
        
        print(f"✅ Received message: '{user_message}'. Processing for session: {session_id}")

        # Start the long-running task in a background thread
        thread = threading.Thread(
            target=process_in_background, 
            args=(user_message, session_id)
        )
        thread.start()

        # Immediately return an empty 200 OK to Google to satisfy the timeout.
        return jsonify({}), 200

    except Exception as e:
        print(f"An error occurred in the chat endpoint: {e}")
        traceback.print_exc()
        return jsonify({}), 500
# --- Main Application Entry Point ---
if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    day_of_week = os.getenv("PROACTIVE_DAY_OF_WEEK", "fri")
    hour = int(os.getenv("PROACTIVE_HOUR", 11))
    
    scheduler.add_job(trigger_proactive_suggestion, 'cron', day_of_week=day_of_week, hour=hour)
    scheduler.start()
    print(f"✅ Proactive suggestion job scheduled for every {day_of_week.capitalize()} at {hour}:00.")
    
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 8080))

    print("🚀 Starting Funky Travel Bot Server...")
    print(f"🔗 Listening on http://{FLASK_HOST}:{FLASK_PORT}")
    
    app.run(host=FLASK_HOST, port=FLASK_PORT)