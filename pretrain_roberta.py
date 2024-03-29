#!/usr/bin/env python
# coding: utf-8

import sys
from transformers import (
    RobertaForMaskedLM,
    RobertaConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, DatasetDict
from utils import read_conllu, DataCollatorForCharLanguageModeling
import numpy as np
import evaluate

# -----------------------------------------------------------------------------
pretrain_output_dir = "./models/pretrain/test-roberta"
num_train_epochs = 10 

train_path = "train.conllu"
valid_path = "test.conllu"
# token_format can be "subword" or "char". It affects tokenization and word masking.
# You can use a character tokenizer with token_format="subword" but then you won't get whole-word masking.
token_format = "char"
tokenizer_path = "./tokenizers/grc_char/"

mlm_probability = 0.15
max_seq_length = 512
callbacks = []

if len(sys.argv) == 2:
    config_file = sys.argv[1]
    if len(sys.argv) == 2:
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
        
training_arguments = {
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "save_steps": 1000,
    "save_total_limit": 10,
    "logging_steps": 500,
    "eval_steps": 1000,
    "lr_scheduler_type": "constant_with_warmup",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": num_train_epochs,
    "warmup_ratio": 0.1,
    "weight_decay": 0.0,
    "fp16": True,
    "load_best_model_at_end": True,
    "output_dir": pretrain_output_dir,
}

model_config = RobertaConfig(
    num_attention_heads=12,
    num_hidden_layers=12,
    hidden_size=768,
    intermediate_size=3072,
)




# -----------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_space_prefix=True)
model_config.vocab_size = tokenizer.vocab_size
model_config.max_position_embeddings = max_seq_length+1
model_config.pad_token_id = tokenizer.pad_token_id
model_config.bos_token_id = tokenizer.bos_token_id
model_config.eos_token_id = tokenizer.eos_token_id

# Add token_format to the model config so it is saved to "config.json" and can be referenced later for fine-tuning
model_config.token_format = token_format

train_sentences, _ = read_conllu(train_path)
valid_sentences, _ = read_conllu(valid_path)

total_lines = 0
total_split_lines = 0

def tokenize_function(examples):
    global total_lines, total_split_lines

    word_cls_id = tokenizer.get_vocab()['[WORD_CLS]']
    lines = [[tokenizer.encode(token, add_special_tokens=False) for token in line] for line in examples["tokens"]]
    if token_format == 'char':
        lines = [[[word_cls_id] + word[1:] for word in line] for line in lines]

    char_chunk_size = max_seq_length-2
    flattened = []
    for line in lines:
        word_idx = 0
        while word_idx < len(line):
            line_tokens = []
            while word_idx < len(line) and len(line_tokens) + len(line[word_idx]) < char_chunk_size:
                line_tokens.extend(line[word_idx])
                word_idx += 1
            flattened.append(line_tokens)

    total_lines += len(examples["tokens"])
    total_split_lines += len(flattened) - len(examples["tokens"])

    tokenized_inputs = [tokenizer.prepare_for_model(line,
                                return_attention_mask=True,
                                return_token_type_ids=True,
                                return_special_tokens_mask=True) for line in flattened]
    tokenized_inputs = { key: [row[key] for row in tokenized_inputs] for key in tokenized_inputs[0]}
    return tokenized_inputs

# Datasets.map() doesn't work if we split lines (the returned batch must be the same length as the input)
if False:
    train_dataset = Dataset.from_dict({"tokens": train_sentences})
    validation_dataset = Dataset.from_dict({"tokens": valid_sentences})
    datasets = DatasetDict({"train": train_dataset, "validation": validation_dataset})
    tokenized_datasets = datasets.map(tokenize_function, batched=True, desc="Running tokenizer on dataset")
else:
    train_dataset = Dataset.from_dict(tokenize_function({"tokens": train_sentences}))
    validation_dataset = Dataset.from_dict(tokenize_function({"tokens": valid_sentences}))
    tokenized_datasets = DatasetDict({"train": train_dataset, "validation": validation_dataset})

if total_split_lines > 0:
    print(f'Warning: {total_split_lines}/{total_lines} lines were split into multiple lines of max_seq_length={max_seq_length}')

if model_config.token_format == "subword":
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
elif model_config.token_format == "char":
    data_collator = DataCollatorForCharLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
else:
    raise ValueError(f"Unsupported token format '{model_config.token_format}'")

model = RobertaForMaskedLM(model_config)
args = TrainingArguments(**training_arguments)

metric = evaluate.load("accuracy")
def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)

    indices = [[i for i, x in enumerate(labels[row]) if x != -100] for row in range(len(labels))]

    labels = [labels[row][indices[row]] for row in range(len(labels))]
    labels = [item for sublist in labels for item in sublist]

    predictions = [predictions[row][indices[row]] for row in range(len(predictions))]
    predictions = [item for sublist in predictions for item in sublist]

    results = metric.compute(predictions=predictions, references=labels)
    return { "eval_accuracy": results["accuracy"] }

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=callbacks
)
trainer.train()
trainer.save_model(pretrain_output_dir)