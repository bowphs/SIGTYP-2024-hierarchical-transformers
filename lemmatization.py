#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
from typing import List, Tuple
from conllu import parse
from datasets import Dataset, DatasetDict
from datasets import ClassLabel, Value, Features, Sequence
import nltk
import numpy as np
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoConfig,
    pipeline,
)
import torch

assert torch.cuda.is_available()
import os
from pprint import pprint
from datasets import ClassLabel, Sequence
import random
import pandas as pd

import warnings
import evaluate
from sklearn.exceptions import UndefinedMetricWarning
from transformers import (
    EarlyStoppingCallback,
    IntervalStrategy,
    AutoModelForSeq2SeqLM,
    DebertaV2Model,
    RobertaForCausalLM,
    RobertaConfig,
    BertTokenizer,
)

from transformers import EncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_conllu(
    file_path: str, label_name: str
) -> Tuple[List[List[str]], List[List[str]]]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    sentences = []
    labels = []
    for sentence in parse(data):
        tokens = [f"{token['form']} [UPOS_{token['upos']}]" for token in sentence]
        label = [f"{token['lemma']}" for token in sentence]

        sentences.append(tokens)
        labels.append(label)

    return sentences, labels


def postprocess_lemmas(
    sents, labels, separator_token="[SEP]",
):
    postprocessed_sents, postprocessed_labels = [], []
    for sent, label in zip(sents, labels):
        for i in range(len(sent)):
            to_lemmatize = sent[i]
            modified_tokens = to_lemmatize
            postprocessed_sents.append(modified_tokens)
            postprocessed_labels.append(label[i])
    return postprocessed_sents, postprocessed_labels


metric = evaluate.load("exact_match")


identifier = "lat"
tokenizer_path = f"./tokenizers/{identifier}_char/"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_space_prefix=True)
model = AutoModelForSeq2SeqLM.from_pretrained(f"path/to/model")
print(f"Model config:\n{model.config}")

# In[ ]:


train_file = f"data/ST2024/morphology/train/{identifier}_train.conllu"
valid_file = f"data/ST2024/morphology/valid/{identifier}_valid.conllu"
test_file = f"data/ST2024/morphology/test/{identifier}_test.conllu"

train_sentences, train_labels = read_conllu(train_file, "lemma")
valid_sentences, valid_labels = read_conllu(valid_file, "lemma")
test_sentences, test_labels = read_conllu(test_file, "lemma")

train_sentences, train_labels = postprocess_lemmas(train_sentences, train_labels,)
valid_sentences, valid_labels = postprocess_lemmas(valid_sentences, valid_labels,)
test_sentences, test_labels = postprocess_lemmas(test_sentences, test_labels,)


for train_sentence, train_label in zip(train_sentences[:10], train_labels[:10]):
    print(train_sentence)
    print(train_label)
    print("###")

label_list = list(
    set([label for sublist in train_labels + valid_labels for label in sublist])
)

features = Features(
    {"tokens": Value("string"), "labels": Value(dtype="string", id=None)}
)

train_dataset = Dataset.from_dict(
    {"tokens": train_sentences, "labels": train_labels}, features=features
)
valid_dataset = Dataset.from_dict(
    {"tokens": valid_sentences, "labels": valid_labels}, features=features
)
test_dataset = Dataset.from_dict(
    {"tokens": test_sentences, "labels": test_labels}, features=features
)
datasets = DatasetDict(
    {"train": train_dataset, "validation": valid_dataset, "test": test_dataset}
)

labels = datasets["train"].features["labels"]


def compute_metrics(eval_pred: transformers.trainer_utils.EvalPrediction) -> dict:
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    for idx in range(len(decoded_preds)):
        print(f"{decoded_preds[idx]}: {decoded_labels[idx]}")
        if idx == 100:
            break

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"exact_match": result["exact_match"]}
    return {k: round(v, 4) for k, v in result.items()}


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, max_length=512
    )  # is_split_into_words=True,
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["labels"], max_length=512, truncation=True)

    for example in examples["tokens"]:
        word, tag = example.split(" [")
        tag = "[" + tag
        tokenized_word = tokenizer(word, add_special_tokens=False)
        tokenized_pair = tokenizer(example, add_special_tokens=False)
        assert (
            len(tokenized_pair.input_ids) == len(tokenized_word.input_ids) + 2
        ), f"Tokenization error: '{example}' did not tokenize as expected. Word token length: {len(tokenized_word.input_ids)}, Pair token length: {len(tokenized_pair.input_ids)}"

    tokenized_inputs["labels"] = labels["input_ids"]
    return tokenized_inputs


print(f"Number of parameters: {model.num_parameters()}")

random_seed = 2
num_train_epochs = 200
gradient_accumulation_steps = 4
batch_size = 32
learning_rate = 1e-5
random_seed = 42
weight_decay = 1e-3

args = Seq2SeqTrainingArguments(
    run_name=f"sigtyp-lemmatization-{identifier}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=int(batch_size / 4),
    weight_decay=weight_decay,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    report_to="wandb",
    generation_max_length=30,
    seed=random_seed,
    generation_num_beams=20,
    load_best_model_at_end=True,
    metric_for_best_model="exact_match",
    save_total_limit=5,
    output_dir=f"path/to/output",
    fp16=True,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
)
tokenized_dataset = datasets.map(tokenize_and_align_labels, batched=True)

save_name = f"sigtyp-lemmatization"
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
)
trainer.train()
trainer.evaluate()
test_results = trainer.predict(tokenized_dataset["test"])
metrics = test_results.metrics
trainer.log(metrics)
num_return_sequences = 3


with open("test.conllu", "r", encoding="utf-8") as file:
    test_content = file.read()
sentences = parse(test_content)[:10]

for ids, sentence in tqdm(enumerate(sentences), total=len(sentences)):
    for idt, token in enumerate(sentence):
        input = f"{sentences[ids][idt]['form']} [UPOS_{sentences[ids][idt]['upos']}]"
        input = tokenizer(input, return_tensors="pt").input_ids
        prediction = model.generate(
            input.to(device),
            num_beams=20,
            num_return_sequences=num_return_sequences,
            max_length=30,
        )
        output = [
            tokenizer.decode(
                prediction[a],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for a in range(num_return_sequences)
        ]
        sentences[ids][idt]["lemma"] = "[BEAM_SEP]".join(output)

with open(
    f"predictions/lemmatisation/lemma_predictions_form+upos_{identifier}.conllu", "w"
) as f:
    f.writelines([sentence.serialize() + "\n" for sentence in sentences])
