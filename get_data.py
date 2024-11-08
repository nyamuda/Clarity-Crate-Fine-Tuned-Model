import pandas as pd
from datasets import Dataset, DatasetDict




def get_training_data() : 
    data = pd.read_csv('./clarity_crate_expansion.csv')
    train_dataset = Dataset.from_pandas(data)
    
    # Create a DatasetDict with only the train split
    dataset_dict = DatasetDict({"train": train_dataset})
    
    # Save the dataset locally
    dataset_dict.save_to_disk("./training_data/")

