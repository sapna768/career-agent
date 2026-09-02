from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

load_dotenv(override=True)
openai = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

def push(message):
    print(f"Push: {message}")

    payload = {
        "user": pushover_user,
        "token": pushover_token,
        "message": message
    }
    requests.post(pushover_url, data=payload)

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}") 
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer") 
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]

[{'type': 'function',
  'function': {'name': 'record_user_details',
   'description': 'Use this tool to record that a user is interested in being in touch and provided an email address',
   'parameters': {'type': 'object',
    'properties': {'email': {'type': 'string',
      'description': 'The email address of this user'},
     'name': {'type': 'string',
      'description': "The user's name, if they provided it"},
     'notes': {'type': 'string',
      'description': "Any additional information about the conversation that's worth recording to give context"}},
    'required': ['email'],
    'additionalProperties': False}}},
 {'type': 'function',
  'function': {'name': 'record_unknown_question',
   'description': "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
   'parameters': {'type': 'object',
    'properties': {'question': {'type': 'string',
      'description': "The question that couldn't be answered"}},
    'required': ['question'],
    'additionalProperties': False}}}]

def handle_tool_calls(tool_calls):
    results = []

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}

        results.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id
        })
    return results

reader = PdfReader("Sapna_Gupta_CV.pdf")
linkedin = ""

#read pdf pages and bring them together
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text


name = "Sapna Gupta"

system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer, say so."

system_prompt += f"\n\n## LinkedIn Profile / CV:\n{linkedin}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."


def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    done = False
    while not done: 
        response = openai.chat.completions.create(model="gemini-3.5-flash-lite", messages=messages, tools=tools)
        finish_reason = response.choices[0].finish_reason
        
        if finish_reason=="tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = handle_tool_calls(tool_calls) 
            messages.append(message)
            messages.extend(results)
        else:
            done = True
    return response.choices[0].message.content

if __name__ == "__main__":
    print("About to launch on port:", os.environ.get("PORT", 7860), flush=True)
    gr.ChatInterface(chat).launch (server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
