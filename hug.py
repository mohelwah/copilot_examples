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




