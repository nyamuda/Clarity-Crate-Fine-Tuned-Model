
from datasets import load_dataset
from transformers import AutoTokenizer, Seq2SeqTrainer,  AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
import sys

app=FastAPI()

#load training data
ds = load_dataset("FiscalNote/billsum")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained("Tatenda/ClarityCrate")
fine_tuned_model.generation_config.pad_token_id = tokenizer.pad_token_id
#set the pad_token to eos_token
tokenizer.pad_token=tokenizer.eos_token

# 1. PROCESS THE TRAINING DATA 

#Inputs: The input consists of a single word prefixed with "Define: " (e.g., "Define: Gravity").
#Targets: The target is the combined definition and example (e.g., "Definition: A force that attracts a body toward the center of the earth. Example: The apple fell due to gravity.").
def process_data(examples):
    
    # Input: Word, e.g., "Gravity"
    inputs = ["summarize: " + word for word in examples["text"]]
    #inputs = ["define: " + str(word) if word is not None else "Word: [UNKNOWN]" for word in examples["word"]]
    # Output: Definition + Example combined
    #targets = ["definition: " + definition for definition in examples["definition"]]


     # Tokenize the inputs
    #model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length', return_tensors="pt")
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length', return_tensors="pt")


    # Tokenize the outputs (labels)
    #labels = tokenizer(targets, max_length=1024, truncation=True, padding='max_length', return_tensors="pt").input_ids
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True, padding='max_length', return_tensors="pt")

    
    #model_inputs["labels"] = labels
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs




# 2. TRAIN THE MODEL

def train_model() :
    # Split the training data into training and validation sets
    train_test_split = ds["train"].train_test_split(test_size=0.2)

    #tokenize the data
    tokenized_train_data=train_test_split["train"].map(process_data, batched=True)
    tokenized_evaluation_data=train_test_split["test"].map(process_data, batched=True)
    smaller_train_dataset=tokenized_train_data.shuffle(seed=42).select(range(50))
    smaller_evaluation_dataset=tokenized_evaluation_data.shuffle(seed=42).select(range(50))

    # set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01
    )

    #create the trainer
    trainer= Seq2SeqTrainer(
         model=model,
         args=training_args,
         train_dataset=smaller_train_dataset,
         eval_dataset=smaller_evaluation_dataset,
         tokenizer=tokenizer
    )

    #train the model
    trainer.train()

    #save the fine-tuned model
    trainer.save_model("./fine-tuned-model")

    #evaluate the model
    eval_results=trainer.evaluate()
    print(eval_results)



# Function to generate definition + example based on a word
def generate_output(input):
    # Load the fine-tuned model from the saved directory
    
  
    # Prepare the input
    input_text = f"summarize: {input}"

    # Tokenize the input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    # Generate the output (e.g., the example sentence)
    outputs = fine_tuned_model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example interaction
input = ds['test'][100]['text']

class SummarizationRequest(BaseModel) :
    text:str

def outputValue():
  #train_model()
  generate_output(input)
  #print(ds["train"][0])
  


if __name__=="__main__":
    pass

@app.post("/summarize")
async def summarize_text(request: SummarizationRequest) :
    output= generate_output(request.text)
    return {"output":output}


@app.get("/")
def read_root():
    return {"Hello": "World"}