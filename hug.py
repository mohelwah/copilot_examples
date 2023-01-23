'''
huggingface transformers train model and query model question answer distilbert-base-uncased model
'''

# Importing libraries
import os
import json
import torch
import random
import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

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
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenizing dataset
def tokenize_function(examples):
    return tokenizer(examples["question"], examples["context"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Setting model
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# trian model
# Setting training arguments
training_args = TrainingArguments(
    "test-squad",
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
)
                                    
# Training model
trainer.train()

# Saving model
trainer.save_model("distilbert-base-uncased")

# Loading model
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Query model
def query(question, context):
    encoding = tokenizer.encode_plus(question, context)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
    return answer

# Testing model
question = "What is the name of the repository ?"
context = "Hugging Face is a French company based in New York and Paris. Its goal is to democratize NLP. Its main product is the library transformers that provides state-of-the-art pre-trained models for Natural Language Processing."
print(query(question, context))

# Output
# ['Hugging', 'Face', 'is', 'a', 'French', 'company', 'based', 'in', 'New', 'York', 'and', 'Paris', '.', 'Its', 'goal', 'is', 'to', 'democratize', 'NLP', '.', 'Its', 'main', 'product', 'is', 'the', 'library', 'transformers', 'that', 'provides', 'state', '-', 'of', '-', 'the', '-', 'art', 'pre', '-', 'trained', 'models', 'for', 'Natural', 'Language', 'Processing', '.']








