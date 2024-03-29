#!/usr/bin/env python
# coding: utf-8
from functools import partial
import os
import sys
from typing import List, Tuple
import warnings

from conllu.models import TokenList
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Value,
    Features,
    Sequence,
)
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from transformers import EarlyStoppingCallback, IntervalStrategy
from DeBERTa.huggingface import *

from utils import (
    read_conllu,
    tokenize_and_align_labels,
    AdaptiveDataCollatorForTokenClassification,
)

# -----------------------------------------------------------------------------
task = "upos"
model_checkpoint = "path/to/model"
output_dir = "path/to/output"

training_arguments = {
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 10,
    "weight_decay": 0.01,
    "push_to_hub": False,
    "metric_for_best_model": "f1",
    "load_best_model_at_end": True,
    "output_dir": output_dir,
}

callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]

do_train = True
do_predict = False

train_path = "train.conllu"
valid_path = "test.conllu"
test_path = "test.conllu"

task = ["upos"]

if len(sys.argv) == 2:
    config_file = sys.argv[1]
    if len(sys.argv) == 2:
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
# -----------------------------------------------------------------------------

train_sentences, train_labels = read_conllu(train_path, task)
valid_sentences, valid_labels = read_conllu(valid_path, task)

label_list = sorted(
    list(
        {
            label
            for sublist in train_labels["upos"] + valid_labels["upos"]
            for label in sublist
        }
    )
)

features = Features(
    {
        "tokens": Sequence(Value("string")),
        "labels": Sequence(ClassLabel(names=label_list)),
    }
)

train_dataset = Dataset.from_dict(
    {"tokens": train_sentences, "labels": train_labels["upos"]}, features=features
)
valid_dataset = Dataset.from_dict(
    {"tokens": valid_sentences, "labels": valid_labels["upos"]}, features=features
)
datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset})

labels = datasets["train"].features["labels"]

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", UndefinedMetricWarning)


tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint, add_prefix_space=True, truncation=True
)
label_list = datasets["train"].features["labels"].feature.names
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

config = AutoConfig.from_pretrained(model_checkpoint)
config.label2id = label2id
config.id2label = id2label

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, config=config)
print(f"Number of parameters: {model.num_parameters()}")
label_all_tokens = False
metric = evaluate.load("seqeval")

tokenized_datasets = datasets.map(
    partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        label_all_tokens=label_all_tokens,
        model_config=config,
    ),
    batched=True,
)


model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(**training_arguments)

data_collator = AdaptiveDataCollatorForTokenClassification(
    tokenizer, model_config=config
)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

if do_train:
    trainer.train()
    trainer.evaluate()
    trainer.save_model(output_dir)

if do_predict:
    if not do_train:
        trainer.model = AutoModelForTokenClassification.from_pretrained(output_dir)

    test_sentences, test_labels = read_conllu(test_path, task)
    test_dataset = Dataset.from_dict(
        {"tokens": test_sentences, "labels": test_labels["upos"]}, features=features
    )
    tokenized_test_dataset = test_dataset.map(
        partial(
            tokenize_and_align_labels,
            tokenizer=tokenizer,
            label_all_tokens=label_all_tokens,
        ),
        batched=True,
    )

    predictions, labels, metrics = trainer.predict(
        tokenized_test_dataset, metric_key_prefix="predict"
    )
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    output_predictions_file = os.path.join(output_dir, "predictions.conllu")
    if trainer.is_world_process_zero():
        conllu_predictions = []
        for test_sentence, prediction in zip(test_sentences, true_predictions):
            conllu_predictions.append(
                TokenList(
                    [
                        {
                            "id": idx + 1,
                            "form": test_word,
                            "xpos": None,
                            "upos": pred,
                            "lemma": None,
                            "feats": None,
                            "head": None,
                            "deprel": None,
                            "deps": None,
                            "misc": None,
                        }
                        for idx, (test_word, pred) in enumerate(
                            zip(test_sentence, prediction)
                        )
                    ]
                )
            )
        with open(output_predictions_file, "w") as f:
            f.writelines(
                [sentence.serialize() + "\n" for sentence in conllu_predictions]
            )
