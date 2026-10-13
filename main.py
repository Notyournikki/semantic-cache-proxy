
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from the .env file")

client = Groq(api_key=api_key)

app = FastAPI(title="Semantic Cache Proxy")

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Semantic Cache Proxy is running"}

@app.post("/v1/chat")
def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": request.query}
            ]
        )

        answer = response.choices[0].message.content

        return {
            "query": request.query,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
