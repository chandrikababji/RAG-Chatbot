import os
from openai import OpenAI
from dotenv import load_dotenv

#loading environment variables here
load_dotenv()

#Getting API key From .env
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

#Intialize Client
client=OpenAI(api_key=OPENAI_API_KEY)

def get_answer(query, chunks):
    """
    Generate an AI-based answer using OpenAI's new API.
    """
    context = "\n\n".join(chunks)
    prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {query}"

    response = client.responses.create(
        model="gpt-4o-mini",  # or "gpt-4o" if you have access
        input=prompt
    )

    return response.output_text
