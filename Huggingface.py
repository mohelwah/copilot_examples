'''
HuggingFace script to train and query abstractive question and answer model 
'''
# HuggingFace script to train and query abstractive question and answer model

# Importing libraries
import os
import json
import torch
import random
import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

# Setting seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Loading dataset
dataset = load_dataset("squad_v2")
metric = load_metric("squad_v2")

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Tokenizing dataset
def tokenize_function(examples):
    return tokenizer(examples["question"], examples["context"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Setting model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Setting training arguments
training_args = TrainingArguments(
    "test-t5",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    save_steps=1000,
    save_total_limit=2,
)

# Setting trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Training model
trainer.train()

# Saving model
trainer.save_model()

# Loading model
model = AutoModelForSeq2SeqLM.from_pretrained("test-t5")

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Setting function to generate answer
def generate_answer(question, context):
    encoding = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    # generate logits
    output = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_length=32
    )

    # retrieve the answer
    answer = tokenizer.decode(output[0])
    return answer

# Setting function to evaluate model
def evaluate(context, question, answer):
    # retrieve the answer
    generated_answer = generate_answer(question, context)

    # compute the metrics
    squad_v2_metric = metric.compute(predictions=[generated_answer], references=[answer])
    return squad_v2_metric

# Setting function to query model
def query_model():
    # setting question and context
    question = input("Enter question: ")
    context = input("Enter context: ")

    # generating answer
    answer = generate_answer(question, context)

    # printing answer
    print("Answer: " + answer)

# Setting function to evaluate model
def evaluate_model():
    # setting question and context
    question = input("Enter question: ")
    context = input("Enter context: ")
    answer = input("Enter answer: ")

    # evaluating model
    score = evaluate(context, question, answer)

    # printing score
    print("Score: " + str(score))

# Setting function to train model
def train_model():
    # Setting training arguments
    training_args = TrainingArguments(
        "test-t5",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        save_steps=1000,
        save_total_limit=2,
    )

    # Setting trainer
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],  
        eval_dataset=tokenized_datasets["validation"],
                                        
        tokenizer=tokenizer,
    )

    # Training model
    trainer.train()

# Setting function to save model
def save_model():
    # Saving model
    trainer.save_model()

# Setting function to load model
def load_model():
    # Loading model
    model = AutoModelForSeq2SeqLM.from_pretrained("test-t5")

    # Loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    
# Setting function to main
def main():
    # Setting choice
    choice = input("Enter choice: ")

    # Checking choice
    if choice == "query":
        query_model()
    elif choice == "evaluate":
        evaluate_model()
    elif choice == "train":
        train_model()
    elif choice == "save":
        save_model()
    elif choice == "load":
        load_model()
    else:
        print("Invalid choice")

# Calling main  
if __name__ == "__main__":
    main()

