from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub.inference_api import InferenceApi

app = FastAPI()

API_TOKEN = "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"
# Use your custom model hosted on Hugging Face
inference = InferenceApi(repo_id="Tatenda/ClarityCrate", token=API_TOKEN)

class SummarizationRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize_text(request: SummarizationRequest):
    response = inference(f"summarize: {request.text}", params={"max_length": 128})
    return {"output": response[0]["summary_text"]}

@app.get("/")
def read_root():
    return {"Hello": "world"}
