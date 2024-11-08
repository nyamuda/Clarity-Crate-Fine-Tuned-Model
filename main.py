from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub.inference_api import InferenceApi
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Initialize the Inference API
inference = InferenceApi(
    repo_id="nyamuda/extractive-summarization", token=API_TOKEN, task="summarization")


class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 128  # Default max length for the summary
    min_length: int = 50   # Default min length for the summary


@app.post("/summarize")
async def summarize_text(request: SummarizationRequest):
    parameters = {
        "max_length": request.max_length,
        "min_length": request.min_length,
        "num_beams": 4,  # Optional: adjust for better generation quality
        "length_penalty": 2.0  # Optional: adjust penalty to balance length
    }
    response = inference(f"summarize: {request.text}", params=parameters)

    # Get the summarized text
    summary = response[0]["summary_text"]

    return {"summary": summary}


@app.get("/")
def read_root():
    return "WELCOME"


if __name__ == "__main__":
    pass
