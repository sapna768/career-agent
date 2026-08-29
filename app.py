from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
import os

load_dotenv(override=True)

name = "Sapna Gupta"

openai = OpenAI(
    api_key=os.getenv("google_api_key"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

reader = PdfReader("me/Sapna_Gupta_CV.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

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
    response = openai.chat.completions.create(model="gemini-3.5-flash", messages=messages)
    return response.choices[0].message.content


if __name__ == "__main__":
    gr.ChatInterface(chat).launch()
