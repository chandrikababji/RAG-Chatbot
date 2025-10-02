from openai import OpenAI
from dotenv import load_dotenv
import os

#loading env variables here
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KRY")
#iniatializing the openai client
client=OpenAI(api_key=OPENAI_API_KEY)

def get_answer(query, context):
    prompt = f"Answer the question using the context:\nContext: {context}\nQuestion: {query}\nAnswer:"
    response = OpenAI.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()
