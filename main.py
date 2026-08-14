import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

app = FastAPI(title="App AI Module")

# Allow requests from your mobile/web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client using server environment variable
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

class PromptPayload(BaseModel):
    prompt: str
    task: str | None = "general"

@app.get("/")
def root():
    return {"status": "online", "message": "ML module is operational"}

@app.post("/predict")
def predict(data: PromptPayload):
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")
    try:
        # Customize behavior based on task
        system_instruction = (
            "You are an embedded AI feature inside an app. Be concise, fast, and structured."
            if data.task == "general"
            else f"You are a specialized AI module for {data.task}."
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=data.prompt,
            config={"system_instruction": system_instruction}
        )
        return {"result": response.text, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
