#!/usr/bin/env python
# coding: utf-8

import sys
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)
import os
import models
import torch
import pickle
import tqdm
import numpy as np
from utils import read_conllu, AdaptiveDataCollatorForTokenClassification
from collections import defaultdict

# -----------------------------------------------------------------------------
model_checkpoint = './models/pretrain/test-ctw/'
batch_size = 1
output_dir = "./embeddings/"
identifier = "grc"
save_gensim_format = True
hidden_layer_index = 3

if len(sys.argv) == 2:
    config_file = sys.argv[1]
    if len(sys.argv) == 2:
        with open(config_file) as f:
            print(f.read())
            exec(open(config_file).read())

input_files = [
    f"data/ST2024/morphology/train/{identifier}_train.conllu",
    f"data/ST2024/morphology/valid/{identifier}_valid.conllu",
    #f"data/ST2024/morphology/test/{identifier}_test.conllu",
]
output_file = os.path.join(output_dir, f"{identifier}.bin")
# -----------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

sentences = [sent for file in input_files for sent in read_conllu(file)[0]]
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True, truncation=True)
config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint, config=config).to(device)
model.eval()

print(f"Number of parameters: {model.num_parameters()}")

data_collator = AdaptiveDataCollatorForTokenClassification(tokenizer, model_config=config)

def tokenize(examples, tokenizer, model_config, split_long_words=True):
    word_cls_id = tokenizer.get_vocab()['[WORD_CLS]']

    tokenized_inputs = {
        'input_ids': [],
        'word_token_indices': [],
        'char_input_mask': [],
        'word_input_mask': [],
    }

    for tokens in examples['tokens']:
        token_ids = [tokenizer.encode(token, add_special_tokens=False) for token in tokens]

        assert (token_ids[0][0] == tokenizer.get_vocab()['▁'])
        token_ids = [ids[1:] for ids in token_ids]

        if split_long_words:
            input_ids = []
            word_token_indices = []
            for ids in token_ids:
                word_token_indices.append([])
                for i in range(0, len(ids), model_config.max_word_length-1):
                    word_token_indices[-1].append(len(input_ids))
                    input_ids.append([word_cls_id] + ids[i: i+model_config.max_word_length-1])
        else:
            input_ids = [[word_cls_id] + ids[0:model_config.max_word_length-1] for ids in token_ids]
            word_token_indices = [[i] for i in range(len(input_ids))]

        input_ids = [[word_cls_id, tokenizer.cls_token_id], *input_ids, [word_cls_id, tokenizer.sep_token_id]]
        tokenized_inputs['char_input_mask'].append(
            [[1]*len(ids)+[0]*(model_config.max_word_length-len(ids)) for ids in input_ids])
        input_ids = [ids + [0]*(model_config.max_word_length-len(ids)) for ids in input_ids]

        tokenized_inputs['input_ids'].append(input_ids)
        tokenized_inputs['word_token_indices'].append(word_token_indices)
        tokenized_inputs['word_input_mask'].append([1]*len(input_ids))

    return tokenized_inputs

word_embeds = defaultdict(lambda: torch.zeros(model.config.inter_word_encoder.hidden_size, device=device))
word_counts = defaultdict(int)

with torch.no_grad():
    for sent_index in tqdm.trange(0, len(sentences), batch_size):
        tokens = sentences[sent_index:sent_index+batch_size]
        tokenized_examples = tokenize({"tokens": tokens}, tokenizer, model_config=config)
        word_token_indices = tokenized_examples["word_token_indices"]
        del tokenized_examples["word_token_indices"]
        examples = [{ k: v[i] for k, v in tokenized_examples.items() } for i in range(len(tokens))]
        batch = data_collator.torch_call(examples)
        batch = { k: v.to(device) for k, v in batch.items() }
        embeds = model(**batch, output_hidden_states=True).hidden_states[hidden_layer_index]
        for i, sent in enumerate(tokens):
            for token_index, token in enumerate(sent):
                word_counts[token] += 1
                word_split_embeds = embeds[i][word_token_indices[i][token_index]]
                word_embed = torch.mean(word_split_embeds, dim=0)
                word_embeds[token] += word_embed

    for word in word_embeds:
        word_embeds[word] = (word_embeds[word] / word_counts[word]).cpu().numpy()

print("Number of words:", len(word_embeds))
print("Saving embeddings to", output_file)
os.makedirs(output_dir, exist_ok=True)

if save_gensim_format:
    from gensim.models import KeyedVectors
    model = KeyedVectors(
        vector_size=model.config.inter_word_encoder.hidden_size)
    model.add_vectors(list(word_embeds.keys()), list(word_embeds.values()))
    model.save_word2vec_format(output_file, binary=True)
else:
    pickle.dump(dict(word_embeds), open(output_file, "wb"))
