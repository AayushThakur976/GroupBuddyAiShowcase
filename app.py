import os
import uuid
import traceback
import requests
import threading
import time
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

# --- LOCKING MECHANISM SETUP ---
# This dictionary will hold a lock for each active session_id
SESSION_LOCKS = {}

# --- Proactive Messaging Functions ---
def trigger_proactive_suggestion():
    """
    This function is called by the scheduler to post a travel suggestion.
    """
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
    """
    Posts a given message to the group chat using the configured webhook URL.
    """
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


# --- Core AI Logic Function ---
def run_graph_and_get_response(user_message, session_id):
    """
    Invokes the agent graph and returns the final response.
    """
    print(f"🤖 Running graph for session: {session_id}")
    try:
        config = {"configurable": {"thread_id": session_id}}
        graph_input = {"messages": [HumanMessage(content=user_message)]}
        final_state = travel_agent_graph.invoke(graph_input, config=config)
        print("   - Graph invocation complete.")

        agent_response = "I'm not sure how to respond to that, buddy."
        if messages := final_state.get("messages"):
            for message in reversed(messages):
                if isinstance(message, AIMessage) and not message.tool_calls:
                    agent_response = message.content
                    break
        
        print(f"\n✅ FINAL RESPONSE GENERATED:\n---\n{agent_response}\n---\n")
        return agent_response

    except Exception as e:
        print("\n" + "="*50)
        print(">>> ❌ ERROR: An exception occurred in the graph! <<<")
        print(f"Error Details: {e}")
        print("="*50 + "\n")
        traceback.print_exc()
        return "Sorry, I ran into a problem while thinking about that."

# --- Background Processing now handles the lock ---
def process_in_background(user_message, session_id, lock):
    """
    This function runs in a separate thread for async mode.
    It ensures the lock is released when processing is complete.
    """
    try:
        with app.app_context():
            agent_response = run_graph_and_get_response(user_message, session_id)
            post_message_to_group(agent_response)
    finally:
        # CRUCIAL: Always release the lock when done
        lock.release()
        print(f"🔓 Lock released for session: {session_id}")


# --- API Endpoint now uses the lock ---
@app.route("/api/travel/chat", methods=["POST"])
def chat():
    """
    Handles chat messages, now with a locking mechanism to prevent race conditions.
    """
    try:
        event_data = request.get_json()
        user_message, session_id = "", ""

        # Parsing logic for Google Chat and simple JSON
        if "chat" in event_data and "messagePayload" in event_data["chat"]:
            message_payload = event_data["chat"]["messagePayload"]
            message_field = message_payload.get("message", {})
            if isinstance(message_field, dict):
                user_message = message_field.get("argumentText", "").strip() or message_field.get("text", "").strip()
            session_id = message_payload.get("space", {}).get("name", "")
        else:
            message_field = event_data.get("message", "")
            if isinstance(message_field, str):
                user_message = message_field.strip()
            session_id = event_data.get("session_id", "")

        if not user_message: return jsonify({})
        if not session_id: session_id = f"test-session-{uuid.uuid4()}"
        
        print(f"✅ Received message: '{user_message}'.")
        
        lock = SESSION_LOCKS.setdefault(session_id, threading.Lock())

        if not lock.acquire(blocking=False):
            print(f"🔒 Request for session {session_id} ignored: already processing.")
            return jsonify({}), 200 # Return empty OK to ignore
        
        print(f"🔐 Lock acquired for session: {session_id}")
        
        EXECUTION_MODE = os.getenv("EXECUTION_MODE", "async")

        if EXECUTION_MODE == "sync":
            print("   - Running in SYNC mode.")
            try:
                agent_response = run_graph_and_get_response(user_message, session_id)
                return jsonify({"response": agent_response}), 200
            finally:
                lock.release()
                print(f"🔓 Lock released for session: {session_id}")
        else: # async mode
            print("   - Running in ASYNC mode.")
            thread = threading.Thread(
                target=process_in_background, 
                args=(user_message, session_id, lock)
            )
            thread.start()
            # Return a completely empty 200 OK for the async handshake.
            return jsonify({}), 200

    except Exception as e:
        print(f"An error occurred in the chat endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- Main Application Entry Point ---
if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    day_of_week = os.getenv("PROACTIVE_DAY_OF_WEEK", "*")
    hour = int(os.getenv("PROACTIVE_HOUR", 15))
    minute = int(os.getenv("PROACTIVE_MINUTE", 48))
    
    scheduler.add_job(trigger_proactive_suggestion, 'cron', day_of_week=day_of_week, hour=hour, minute=minute)
    scheduler.start()
    print(f"✅ Proactive suggestion job scheduled for day(s) '{day_of_week}' at {hour}:{minute:02d}.")
    
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 8080))

    print("🚀 Starting Funky Travel Bot Server...")
    print(f"🔗 Listening on http://{FLASK_HOST}:{FLASK_PORT}")
    
    app.run(host=FLASK_HOST, port=FLASK_PORT)