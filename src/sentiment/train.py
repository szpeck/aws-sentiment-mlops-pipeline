# src/sentiment/train.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import evaluate

def train_model():
    # Load IMDB dataset
    dataset = load_dataset("imdb")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    # Tokenize function
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)
    
    # Prepare datasets
    tokenized_train = dataset["train"].map(tokenize, batched=True)
    tokenized_test = dataset["test"].map(tokenize, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )
    
    # Train
    trainer.train()
    
    return model, tokenizer

if __name__ == "__main__":
    train_model()