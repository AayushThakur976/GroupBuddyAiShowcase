import os
import uuid
import traceback
import requests
import threading
import re  # <-- FIX 1: Import for the parser
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# --- Load Environment Variables ---
load_dotenv()

# --- Custom Travel Bot Import ---
from travel_bot_graph import travel_agent_graph, llm

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- LOCKING MECHANISM SETUP ---
SESSION_LOCKS = {}

# --- FIX 1: Parser Function to Clean Markdown ---
def clean_markdown(text: str) -> str:
    """
    A simple parser to remove common markdown formatting.
    """
    text = re.sub(r'#+\s', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'[\*\-]\s', '', text)
    text = re.sub(r'---', '', text)
    text = re.sub(r'\n{2,}', '\n', text.strip())
    return text

# --- Proactive Messaging Functions ---
def trigger_proactive_suggestion():
    with app.app_context():
        # ... (This function is correct)
        print(f"[{datetime.now()}] 🚀 Running proactive suggestion job...")
        today = datetime.now().strftime("%A, %B %d, %Y")
        prompt = f"""You are a fun, friendly travel buddy..."""
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            suggestion = response.content
            post_message_to_group(suggestion)
        except Exception as e:
            print(f"Error during proactive suggestion: {e}")

def post_message_to_group(message: str):
    # ... (This function is correct)
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


# --- FIX 2: Core AI Logic Function (Corrected Version) ---
def run_graph_and_get_response(user_message, session_id):
    """
    Invokes the agent graph and returns the final response.
    """
    print(f"🤖 Running graph for session: {session_id}")
    try:
        config = {"configurable": {"thread_id": session_id}}
        graph_input = {"messages": [HumanMessage(content=user_message)]}
        # Using .invoke() as in your working version
        final_state = travel_agent_graph.invoke(graph_input, config=config)
        print("   - Graph invocation complete.")

        agent_response = "I'm not sure how to respond to that, buddy."
        # This new logic correctly finds the final message
        if messages := final_state.get("messages"):
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                agent_response = last_message.content
            elif isinstance(last_message, ToolMessage):
                agent_response = last_message.content
        
        print(f"\n✅ FINAL RESPONSE GENERATED (RAW):\n---\n{agent_response}\n---\n")
        return agent_response
    except Exception as e:
        print(f"\n{'='*50}\n>>> ❌ ERROR: An exception occurred in the graph! <<<\nError Details: {e}\n{'='*50}\n")
        traceback.print_exc()
        return "Sorry, I ran into a problem while thinking about that."

# --- Background Processing for Async Mode ---
def process_in_background(user_message, session_id, lock):
    try:
        with app.app_context():
            agent_response = run_graph_and_get_response(user_message, session_id)
            cleaned_response = clean_markdown(agent_response) # <-- FIX 1
            print(f"✅ PARSED RESPONSE (CLEAN):\n---\n{cleaned_response}\n---\n")
            post_message_to_group(cleaned_response)
    finally:
        lock.release()
        print(f"🔓 Lock released for session: {session_id}")


# --- API Endpoint ---
@app.route("/api/travel/chat", methods=["POST"])
def chat():
    # ... (This function is correct and uses the functions above)
    try:
        event_data = request.get_json()
        user_message, session_id = "", ""

        if "chat" in event_data and "messagePayload" in event_data["chat"]:
            # ...
            message_payload = event_data["chat"]["messagePayload"]
            message_field = message_payload.get("message", {})
            if isinstance(message_field, dict):
                user_message = message_field.get("argumentText", "").strip() or message_field.get("text", "").strip()
            session_id = message_payload.get("space", {}).get("name", "")
        else:
            # ...
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
            return jsonify({}), 200
        
        print(f"🔐 Lock acquired for session: {session_id}")
        
        EXECUTION_MODE = os.getenv("EXECUTION_MODE", "async")

        if EXECUTION_MODE == "sync":
            print("   - Running in SYNC mode.")
            try:
                agent_response = run_graph_and_get_response(user_message, session_id)
                cleaned_response = clean_markdown(agent_response) # <-- FIX 1
                print(f"✅ PARSED RESPONSE (CLEAN):\n---\n{cleaned_response}\n---\n")
                return jsonify({"response": cleaned_response}), 200
            finally:
                lock.release()
                print(f"🔓 Lock released for session: {session_id}")
        else: # async mode
            # ...
            print("   - Running in ASYNC mode.")
            thread = threading.Thread(
                target=process_in_background, 
                args=(user_message, session_id, lock)
            )
            thread.start()
            return jsonify({}), 200

    except Exception as e:
        print(f"An error occurred in the chat endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- Main Application Entry Point ---
if __name__ == "__main__":
    # ... (This part is correct)
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