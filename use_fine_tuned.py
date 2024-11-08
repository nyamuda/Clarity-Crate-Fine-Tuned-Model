from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load the fine-tuned model and tokenizer only once
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("./fine-tuned-model/")
#fine_tuned_model.generation_config.pad_token_id = tokenizer.pad_token_id

# Set up the pipeline with optional device argument (0 for GPU, -1 for CPU)
generation_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)


def use_fine_tuned(text) :
    output = generation_pipeline(
        f"summarize: {text}",
        max_length=128,
        num_beams=2,  # Try reducing to 2 beams or using greedy decoding
        early_stopping=True
    )
    return {"output": output[0]["generated_text"]}

