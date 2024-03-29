#!/usr/bin/env python
# coding: utf-8
from ast import literal_eval
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
    Features,
    Value,
    Sequence,
)
import evaluate
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
import torch
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    IntervalStrategy,
    TrainingArguments,
    Trainer,
    AutoConfig,
)

from models import (
    MultitaskRobertaForTokenClassification,
    MultitaskDebertaForTokenClassification,
    MultitaskCharToWordDebertaForTokenClassification,
)
from utils import (
    read_conllu,
    tokenize_and_align_labels,
    AdaptiveDataCollatorForTokenClassification,
)


def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result:
                result[key].add(value)
            else:
                result[key] = {"empty", value}
    result = {key: list(sorted(value)) for key, value in result.items()}
    return result


def create_component_dict(id2label, morphology_features):
    id2components = {}
    for key in id2label.keys():
        labels = literal_eval(id2label[key])
        components = [
            labels.get(component, "empty")
            if labels
            else ("empty") * len(morphology_features)
            for component in morphology_features
        ]
        components = [
            label_dict[morphology_features[idx]].index(component)
            if component in label_dict[morphology_features[idx]]
            else label_dict[morphology_features[idx]].index("empty")
            for idx, component in enumerate(components)
        ]
        id2components[key] = torch.tensor(components)
    return id2components


# -----------------------------------------------------------------------------
identifier = "grc"
vocab_path = f"./tokenizers/{identifier}_char/"
model_checkpoint = (
    "path/to/model"
)
batch_size = 4
output_dir = "path/to/output"

do_train = True
do_predict = True
identifier = ""
train_path = f"data/ST2024/morphology/train/{identifier}_train.conllu"
valid_path = f"data/ST2024/morphology/valid/{identifier}_valid.conllu"
test_path = valid_path

num_train_epochs = 50

if len(sys.argv) == 2:
    config_file = sys.argv[1]
    if len(sys.argv) == 2:
        with open(config_file) as f:
            print(f.read())
            exec(open(config_file).read())


training_arguments = {
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": batch_size,
    "per_device_eval_batch_size": batch_size,
    "num_train_epochs": 30,
    "weight_decay": 0.01,
    "push_to_hub": False,
    "metric_for_best_model": "f1",
    "load_best_model_at_end": True,
    "output_dir": output_dir,
    "remove_unused_columns": False,
    "save_total_limit": 1,
}

tasks = ["feats"]
# -----------------------------------------------------------------------------


train_sentences, train_labels = read_conllu(train_path, tasks)
valid_sentences, valid_labels = read_conllu(valid_path, tasks)

literal_values = [
    literal_eval(label)
    for sublist in train_labels["feats"] + valid_labels["feats"]
    for label in sublist
]
literal_values = list(filter(lambda item: item is not None, literal_values))
label_dict = merge_dicts(*literal_values)

label_list = sorted(
    list(
        {
            label
            for sublist in train_labels["feats"] + valid_labels["feats"]
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
    {"tokens": train_sentences, "labels": train_labels["feats"]}, features=features
)
valid_dataset = Dataset.from_dict(
    {"tokens": valid_sentences, "labels": valid_labels["feats"]}, features=features
)
datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset})
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", UndefinedMetricWarning)

tokenizer = AutoTokenizer.from_pretrained(
    vocab_path, add_prefix_space=True, truncation=True
)
label_list = datasets["train"].features["labels"].feature.names
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

id2components = create_component_dict(id2label, list(label_dict.keys()))
components2id = {value: key for key, value in id2components.items()}


config = AutoConfig.from_pretrained(model_checkpoint)
config.label2id = label2id
config.id2label = id2label

if config.model_type == "roberta":
    model = MultitaskRobertaForTokenClassification.from_pretrained(
        model_checkpoint,
        config=config,
        labels_dict=label_dict,
        id2components=id2components,
    )
elif config.model_type == "deberta-v2":
    model = MultitaskDebertaForTokenClassification.from_pretrained(
        model_checkpoint,
        config=config,
        labels_dict=label_dict,
        id2components=id2components,
    )
elif config.model_type == "ctw_deberta":
    model = MultitaskCharToWordDebertaForTokenClassification.from_pretrained(
        model_checkpoint,
        config=config,
        labels_dict=label_dict,
        id2components=id2components,
    )
else:
    raise ValueError("Model type not supported")

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


args = TrainingArguments(**training_arguments)

tokenized_datasets = tokenized_datasets.remove_columns(["tokens"])
data_collator = AdaptiveDataCollatorForTokenClassification(
    tokenizer, model_config=config
)


def predict_morphology(predictions, labels, morphology_features):
    memoize = {}
    predictions = {
        component: torch.argmax(torch.tensor(predictions[component]), axis=2)
        for component in predictions.keys()
    }
    predictions = torch.stack(
        [predictions[component] for component in predictions.keys()], axis=-1
    )
    sentences_predictions, sentences_labels = [], []
    for prediction, label in zip(predictions, labels):
        tokens_predictions, tokens_labels = [], []
        for pred, lab in zip(prediction, label):
            if lab != -100 and lab != len(id2components):
                lab = id2components[lab]
                if lab in memoize:
                    token_labels = memoize[lab]
                else:
                    token_labels = "|".join(
                        [
                            f"{morphology_features[idx]}={label_dict[morphology_features[idx]][pr]}"
                            for idx, pr in enumerate(lab)
                        ]
                    )
                    memoize[lab] = token_labels
                if pred in memoize:
                    token_predictions = memoize[pred]
                else:
                    token_predictions = "|".join(
                        [
                            f"{morphology_features[idx]}={label_dict[morphology_features[idx]][pr]}"
                            for idx, pr in enumerate(pred)
                        ]
                    )
                    memoize[pred] = token_predictions
                tokens_predictions.append(token_predictions)
                tokens_labels.append(token_labels)
        sentences_predictions.append(tokens_predictions)
        sentences_labels.append(tokens_labels)
    return sentences_predictions, sentences_labels


def compute_metrics(p):
    predictions, labels = p
    sentences_predictions, sentences_labels = predict_morphology(
        predictions, labels, list(label_dict.keys())
    )
    results = metric.compute(
        predictions=sentences_predictions, references=sentences_labels
    )
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

if do_train:
    trainer.train()
    trainer.evaluate()
    trainer.save_model(output_dir)

if do_predict:
    if not do_train:
        trainer.model = MultitaskRobertaForTokenClassification.from_pretrained(
            output_dir, labels_dict=label_dict, id2components=id2components
        )

    test_sentences, test_labels = read_conllu(test_path, tasks)
    test_dataset = Dataset.from_dict(
        {"tokens": test_sentences, "labels": test_labels["feats"]}, features=features
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

    true_predictions, true_labels = predict_morphology(
        predictions, labels, list(label_dict.keys())
    )

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
                            "upos": None,
                            "lemma": None,
                            "feats": {
                                key: value
                                for p in pred.split("|")
                                for key, value in [p.split("=")]
                                if value != "empty"
                            },
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
