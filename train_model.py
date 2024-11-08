
from datasets import load_dataset
from transformers import AutoTokenizer, Seq2SeqTrainer,  AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments


#load training data
#ds = load_dataset("FiscalNote/billsum")
ds = load_dataset("ccdv/pubmed-summarization", "document")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
#set the pad_token to eos_token
tokenizer.pad_token=tokenizer.eos_token

# 1. PROCESS THE TRAINING DATA 
def process_data(examples):
    
    # Input: Word, e.g., "Gravity"
    inputs = ["summarize: " + text for text in examples["article"]]

    #Tokenize the inputs
    #model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length', return_tensors="pt")
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length', return_tensors="pt")

    # Tokenize the outputs (labels)
    labels = tokenizer(text_target=examples["abstract"], max_length=128, truncation=True, padding='max_length', return_tensors="pt")

    #model_inputs["labels"] = labels
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs




#7 2. TRAIN THE MODEL
def train_model() :
   

    # Split the training data into training and validation sets
    #train_test_split = ds["train"].train_test_split(test_size=0.2)

    #tokenize the data
    tokenized_train_data=ds["train"].map(process_data, batched=True)
    tokenized_evaluation_data=ds["test"].map(process_data, batched=True)
    smaller_train_dataset=tokenized_train_data.shuffle(seed=42).select(range(1000))
    smaller_evaluation_dataset=tokenized_evaluation_data.shuffle(seed=42).select(range(100))

    # set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
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

