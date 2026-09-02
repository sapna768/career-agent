from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
import json
import os
import requests

load_dotenv(override=True)

print("Starting app...", flush=True)

name = "Sapna Gupta"

openai = OpenAI(
    api_key=os.getenv("google_api_key"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

reader = PdfReader("me/Sapna_Gupta_CV.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

print("PDF loaded, starting Gradio...", flush=True)

system_prompt = f"""You are acting as {name}. You are answering questions on {name}'s website,
particularly questions related to {name}'s career, background, skills and experience.
Your responsibility is to represent {name} for interactions on the website as faithfully as possible.
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions.
Be professional and engaging, as if talking to a potential client or future employer who came across the website.

## LinkedIn Profile / CV:
{linkedin}

With this context, please chat with the user, always staying in character as {name}.

## Tools — follow this exactly

You have two tools:

1. record_unknown_question — use this whenever you cannot answer from the CV/context. Never invent facts.
2. record_user_details — use this whenever the visitor shares contact details (email required; name and notes if you have them).

Do not ask for name or email on every message, and not on the first greeting.

Ask for name and email only once, and only if:
- you cannot answer their question, or
- they say they want {name} to contact them.

CRITICAL: If a question asks about something not in the profile, do NOT answer it or make excuses. 
Your VERY FIRST step MUST be to ask the visitor for their email address so {name} can follow up with them. 
Only after they provide an email (or decline) should you call record_unknown_question.

If they already gave name or email in this same chat, do not ask again. Call record_user_details immediately.
If they already filled the visitor form, call record_user_details and do not ask again.
"""


def push(message):
    print(f"Push: {message}", flush=True)
    if not pushover_user or not pushover_token:
        print("Pushover keys missing; skip send", flush=True)
        return
    requests.post(
        pushover_url,
        data={"user": pushover_user, "token": pushover_token, "message": message},
        timeout=15,
    )



def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"New contact: {name}\nEmail: {email}\nNotes: {notes}")
    return "OK"


def record_unknown_question(question, email="not provided", name="not provided"):
    push(
        f"Could not answer this question:\n{question}\n\n"
        f"Visitor name: {name}\nVisitor email: {email}"
    )
    return "OK"


record_user_details_json = {
    "name": "record_user_details",
    "description": (
        "Record that a visitor wants to be contacted. Call this as soon as they "
        "give an email address. Include their name and notes about what they asked."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {
                "type": "string",
                "description": "What they asked or any context worth recording",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": (
        "Always use this tool when you cannot answer a question. Include the "
        "visitor's name and email if they already provided them in the conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
            "email": {
                "type": "string",
                "description": "Visitor email if already given, otherwise omit or use 'not provided'",
            },
            "name": {
                "type": "string",
                "description": "Visitor name if already given, otherwise omit or use 'not provided'",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]

TOOL_FUNCTIONS = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
}

_recorded_emails = set()


def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        print(f"Tool called: {tool_name} args={arguments}", flush=True)
        tool = TOOL_FUNCTIONS.get(tool_name)
        result = tool(**arguments) if tool else "No tool found"
        results.append(
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            }
        )
    return results


def chat(message, history, visitor_name="", visitor_email=""):
    visitor_name = (visitor_name or "").strip()
    visitor_email = (visitor_email or "").strip()
    form_note = ""
    if visitor_name or visitor_email:
        form_note = (
            f"\n\n[Visitor form: name={visitor_name or 'not provided'}, "
            f"email={visitor_email or 'not provided'}]"
        )
        email_key = visitor_email.lower()
        if visitor_email and email_key not in _recorded_emails:
            _recorded_emails.add(email_key)
            record_user_details(
                visitor_email,
                visitor_name or "Name not provided",
                notes=message,
            )

    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": message + form_note}
    ]
    response = openai.chat.completions.create(
        model="gemini-3.5-flash",
        messages=messages,
        tools=tools,
    )
    while response.choices[0].finish_reason == "tool_calls":
        tool_message = response.choices[0].message
        results = handle_tool_calls(tool_message.tool_calls)
        messages.append(tool_message)
        messages.extend(results)
        response = openai.chat.completions.create(
            model="gemini-3.5-flash",
            messages=messages,
            tools=tools,
        )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("About to launch on port:", os.environ.get("PORT", 7860), flush=True)
    gr.ChatInterface(
        chat,
        title=f"Chat with {name}",
        additional_inputs=[
            gr.Textbox(label="Your name", placeholder="Name"),
            gr.Textbox(label="Your email", placeholder="email@example.com")

        ]
    ).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
