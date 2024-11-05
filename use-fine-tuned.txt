from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

# Load the fine-tuned model and tokenizer only once
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("Tatenda/ClarityCrate")

# Set up the pipeline with optional device argument (0 for GPU, -1 for CPU)
generation_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

class SummarizationRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize_text(request: SummarizationRequest):
    # Use the pipeline with streamlined settings
    output = generation_pipeline(
        f"summarize: {request.text}",
        max_length=128,
        num_beams=2,  # Try reducing to 2 beams or using greedy decoding
        early_stopping=True
    )
    return {"output": output[0]["generated_text"]}

@app.get("/")
def read_root():
    return {"Hello": "World"}
