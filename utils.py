import torch
from collections import defaultdict

import random
from typing import List, Tuple, Union, Optional, Any, Dict, Mapping
from conllu import parse
from dataclasses import dataclass
import warnings
from typing import List, Tuple, Iterable
from pprint import pprint

from transformers.data.data_collator import (
    DataCollatorMixin,
    PaddingStrategy,
    DataCollatorForWholeWordMask,
    _torch_collate_batch,
    tolist,
)
from transformers import PreTrainedTokenizerBase

UD_FIELDS = ("upos", "feats", "heads", "deprels", "lemmas")


def read_conllu(
    file_path: str, label_names: Iterable[str] = None
) -> Tuple[List[List[str]], List[List[str]]]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    sentences = []
    labels = defaultdict(list)
    for sentence in parse(data):
        tokens = [token["form"] for token in sentence]
        sentences.append(tokens)

        if label_names is not None:
            for label_name in label_names:
                local_labels = []
                for idx, token in enumerate(sentence):
                    local_labels.append(str(token[label_name]))
                labels[label_name].append(local_labels)
    return sentences, labels


def tokenize_char_to_word(
    examples, tokenizer, label_all_tokens, model_config, split_long_words=True
):
    tokenized_inputs = {
        "input_ids": [],
        "char_input_mask": [],
        "word_input_mask": [],
        "labels": [],
    }
    word_cls_id = tokenizer.get_vocab()["[WORD_CLS]"]
    for tokens, token_labels in zip(examples["tokens"], examples["labels"]):
        token_ids = [
            tokenizer.encode(token, add_special_tokens=False) for token in tokens
        ]

        assert token_ids[0][0] == tokenizer.get_vocab()["▁"]
        token_ids = [ids[1:] for ids in token_ids]

        if split_long_words:
            input_ids, labels = [], []
            for word_id, (ids, label) in enumerate(zip(token_ids, token_labels)):
                for i in range(0, len(ids), model_config.max_word_length - 1):
                    input_ids.append(
                        [word_cls_id] + ids[i : i + model_config.max_word_length - 1]
                    )
                    labels.append(label if i == 0 or label_all_tokens else -100)
        else:
            input_ids = [
                [word_cls_id] + ids[0 : model_config.max_word_length - 1]
                for ids in token_ids
            ]
            labels = token_labels

        # Add [CLS] and [SEP] tokens
        input_ids = [
            [word_cls_id, tokenizer.cls_token_id],
            *input_ids,
            [word_cls_id, tokenizer.sep_token_id],
        ]
        labels = [-100, *labels, -100]

        tokenized_inputs["char_input_mask"].append(
            [
                [1] * len(ids) + [0] * (model_config.max_word_length - len(ids))
                for ids in input_ids
            ]
        )

        # pad words to max_word_length
        input_ids = [
            ids + [0] * (model_config.max_word_length - len(ids)) for ids in input_ids
        ]
        tokenized_inputs["input_ids"].append(input_ids)

        tokenized_inputs["word_input_mask"].append([1] * len(input_ids))
        tokenized_inputs["labels"].append(labels)

    return tokenized_inputs


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens, model_config):
    if label_all_tokens:
        warnings.warn(
            "label_all_tokens is set to True, so each subword is considered in the evaluation."
        )

    token_format = getattr(model_config, "token_format", None)
    if token_format == "char_to_word":
        return tokenize_char_to_word(
            examples, tokenizer, label_all_tokens, model_config
        )

    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512
    )

    if token_format == "char":
        # the tokenizer adds a space before each token; replace it with [WORD_CLS] to be consistent with pre-training
        word_cls_id = tokenizer.get_vocab()["[WORD_CLS]"]
        space_id = tokenizer.get_vocab()["▁"]
        tokenized_inputs["input_ids"] = [
            [word_cls_id if id == space_id else id for id in ids]
            for ids in tokenized_inputs["input_ids"]
        ]

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        word_id_2_first_subword = {
            word_id: word_ids.index(word_id) for word_id in word_ids
        }
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Copied from transformers.DataCollatorForTokenClassification and modified to support CharToWord models
@dataclass
class AdaptiveDataCollatorForTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        model_config (`object`, *optional*):
            Model config.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model_config: Optional[object] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    max_word_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        label_names = [
            feature
            for feature in features[0].keys()
            if feature in UD_FIELDS or feature == "labels"
        ]
        skip_names = ("word_input_mask", *label_names)

        if getattr(self.model_config, "token_format", None) == "char_to_word":
            pad_word = [[0] * self.model_config.max_word_length]
            sequence_length = max(len(feature["input_ids"]) for feature in features)
            batch = {
                k: torch.tensor(
                    [f[k] + pad_word * (sequence_length - len(f[k])) for f in features],
                    dtype=torch.int64,
                )
                for k in features[0]
                if k not in skip_names
            }
            batch["word_input_mask"] = torch.tensor(
                [
                    f["word_input_mask"]
                    + [0] * (sequence_length - len(f["word_input_mask"]))
                    for f in features
                ],
                dtype=torch.int64,
            )
        else:
            no_labels_features = [
                {k: v for k, v in feature.items() if k not in skip_names}
                for feature in features
            ]
            batch = self.tokenizer.pad(
                no_labels_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            sequence_length = batch["input_ids"].shape[1]

        if len(label_names) == 0:
            return batch

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        for label_name in label_names:
            labels = (
                [feature[label_name] for feature in features]
                if label_name in features[0].keys()
                else None
            )
            if self.tokenizer.padding_side == "right":
                batch[label_name] = [
                    to_list(label)
                    + [self.label_pad_token_id] * (sequence_length - len(label))
                    for label in labels
                ]
            else:
                batch[label_name] = [
                    [self.label_pad_token_id] * (sequence_length - len(label))
                    + to_list(label)
                    for label in labels
                ]
            batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        return batch


# Adapted from transformers.DataCollatorForWholeWordMask
@dataclass
class DataCollatorForCharLanguageModeling(DataCollatorForWholeWordMask):
    word_cls: str = "[WORD_CLS]"
    min_words: int = 4

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        padded_inputs = {
            key: _torch_collate_batch(
                [e[key] for e in examples],
                self.tokenizer,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            for key in examples[0]
        }
        mask_labels = []
        for e in examples:
            ref_tokens = [
                self.tokenizer._convert_id_to_token(id) for id in tolist(e["input_ids"])
            ]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _torch_collate_batch(
            mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )
        inputs, labels = self.torch_mask_tokens(padded_inputs["input_ids"], batch_mask)
        padded_inputs["input_ids"] = inputs
        padded_inputs["labels"] = labels
        return padded_inputs

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == self.word_cls:
                cand_indexes.append([])
            elif token not in self.tokenizer.all_special_tokens:
                cand_indexes[-1].append(i)

        # Mask random characters instead of entire words if the number of words is small
        if len(cand_indexes) < self.min_words:
            cand_indexes = [
                [i]
                for i, token in enumerate(input_tokens)
                if token not in self.tokenizer.all_special_tokens
            ]

        random.shuffle(cand_indexes)
        num_to_predict = min(
            max_predictions,
            max(1, int(round(len(input_tokens) * self.mlm_probability))),
        )
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError(
                "Length of covered_indexes is not equal to length of masked_lms."
            )
        mask_labels = [
            1 if i in covered_indexes else 0 for i in range(len(input_tokens))
        ]
        return mask_labels
