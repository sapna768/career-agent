import json
import os
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
from pypdf import PdfReader
import requests

# 1. Environment & API Setup
load_dotenv(override=True)

# Support both variable names
google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key")

if not google_api_key:
    raise ValueError("Google API key missing! Please set GOOGLE_API_KEY in your .env file.")

client = OpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Pushover Notification Setup
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

def push_notification(message: str):
    """Send push notification to mobile via Pushover API."""
    print(f"[PUSH NOTIFICATION]: {message}")
    if pushover_user and pushover_token:
        try:
            payload = {
                "user": pushover_user,
                "token": pushover_token,
                "message": message
            }
            requests.post(pushover_url, data=payload, timeout=5)
        except Exception as e:
            print(f"Failed to send Pushover notification: {e}")
    else:
        print("Pushover credentials missing. Skipping notification send.")

# 2. Agent Tools Definition
def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided"):
    push_notification(f" NEW INTEREST RECORDED:\nName: {name}\nEmail: {email}\nNotes: {notes}")
    return {"status": "success", "message": "User interest details successfully logged."}

def record_unknown_question(question: str):
    push_notification(f"UNANSWERED QUESTION DETECTED:\nQuestion: '{question}'")
    return {"status": "success", "message": "Unknown question reported to host."}

tool_record_user_details = {
    "name": "record_user_details",
    "description": "Use this tool whenever a user provides their email address or contact details to stay in touch.",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address provided by the user."
            },
            "name": {
                "type": "string",
                "description": "The user's name, if provided."
            },
            "notes": {
                "type": "string",
                "description": "Relevant context or notes from the discussion."
            }
        },
        "required": ["email"]
    }
}

tool_record_unknown_question = {
    "name": "record_unknown_question",
    "description": "STRICT MANDATE: Use this tool whenever you cannot answer a user's question due to lack of information, missing details in CV, or off-topic queries.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The exact question asked by the user that could not be answered."
            }
        },
        "required": ["question"]
    }
}

tools = [
    {"type": "function", "function": tool_record_user_details},
    {"type": "function", "function": tool_record_unknown_question}
]

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except Exception:
            arguments = {}

        print(f"[TOOL EXECUTED]: {tool_name} with args {arguments}")

        if tool_name == "record_user_details":
            result = record_user_details(**arguments)
        elif tool_name == "record_unknown_question":
            result = record_unknown_question(**arguments)
        else:
            result = {"error": f"Unknown function {tool_name}"}

        results.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id
        })
    return results

# 3. PDF Parsing & Background Context
linkedin_cv_text = ""
cv_path = "me/Sapna_Gupta_CV.pdf"

if os.path.exists(cv_path):
    try:
        reader = PdfReader(cv_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                linkedin_cv_text += extracted + "\n"
        print(f"Successfully loaded CV from {cv_path}")
    except Exception as e:
        print(f"Error loading CV: {e}")
else:
    print(f"Warning: {cv_path} not found. Running without CV context.")

name = "Sapna Gupta"

system_prompt = f"""You are acting as {name}. You are representing yourself on your personal website and answering questions about your career, background, skills, and experience.

STRICT INSTRUCTIONS:
1. Always stay in character as {name}. Be professional, warm, engaging, and articulate.
2. Base your knowledge strictly on your background/CV detailed below.
3. IF A USER ASKS A QUESTION YOU CANNOT ANSWER (or if the information is not in your CV/background):
   - You MUST immediately execute the `record_unknown_question` tool with the question.
   - Politeness statement: Politely inform the user that you don't have that detail on hand, but you've noted it down to follow up.
4. IF A USER SHARES THEIR EMAIL OR EXPRESSES INTEREST IN CONNECTING:
   - You MUST execute the `record_user_details` tool to capture their email, name, and discussion summary.
5. Continuously try to steer meaningful interactions towards getting in touch via email.

## Context (CV / LinkedIn Profile):
{linkedin_cv_text if linkedin_cv_text else "No CV context provided."}
"""

# 4. Gradio Interface Handler
def chat(message: str, history: list):
    # Prepare OpenAI format messages history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Process Gradio chat history into standard OpenAI messages list
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            messages.append({"role": "user", "content": item[0]})
            messages.append({"role": "assistant", "content": item[1]})
        elif isinstance(item, dict):
            messages.append(item)

    messages.append({"role": "user", "content": message})

    done = False
    model_name = "gemini-3.5-flash"

    while not done:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason

        if finish_reason == "tool_calls" and choice.message.tool_calls:
            assistant_msg = choice.message
            messages.append(assistant_msg)
            tool_results = handle_tool_calls(assistant_msg.tool_calls)
            messages.extend(tool_results)
        else:
            done = True
            return choice.message.content

# 5. Launch App
if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat,
        title=f"Chat with {name}'s AI Agent",
        description=f"Ask questions about {name}'s career, projects, and expertise.",
        
    )
    demo.launch(share=True)