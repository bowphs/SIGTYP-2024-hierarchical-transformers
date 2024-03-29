#!/usr/bin/env python
# coding: utf-8

import sys
import os
import conllu
from dataclasses import dataclass, asdict
from DeBERTa.train import train
from DeBERTa.train.deberta.spm_tokenizer import SPMTokenizer

# -----------------------------------------------------------------------------
task = "pretrain"
train_path = "train.conllu"
valid_path = "test.conllu"
pretrain_output_dir = "./models/pretrain/test-deberta"
num_train_epochs = 10
vocab_path = "./tokenizers/grc_char/spm.model"
identifier = ""
if len(sys.argv) == 2:
    config_file = sys.argv[1]
    if len(sys.argv) == 2:
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())

@dataclass
class DebertaTrainingArgs:
    model_config: str = './DeBERTa/configs/ctw_rtd_base.json'
    task_name: str = "rtd" # mlm, rtd
    do_train: bool = True
    max_seq_length: int = 512
    max_word_length: int = 16
    data_dir: str = os.path.join(pretrain_output_dir, "data")
    vocab_path: str = f"./tokenizers/{identifier}_char/spm.model"
    output_dir: str = pretrain_output_dir
    token_format: str = "char_to_word" # subword, char, char_to_word
    num_train_epochs: int = num_train_epochs
    vocab_type: str = "spm"
    fp16: bool = True
    log_steps: int = 250
    dump_interval: int = 10000
    warmup_proportion: float = 0.1
    learning_rate: float = 1e-5
    seed: int = 1234
    max_ngram: int = 4 # max number of consecutive words to mask if whole_word_mask is true, or characters if whole_word_mask is false
    whole_word_mask: bool = False
    train_batch_size: int = 16
    eval_batch_size: int = 32

args = DebertaTrainingArgs()


# -----------------------------------------------------------------------------

def encode(input_file, output_file, max_seq_length, max_word_length, token_format, tokenizer, split_long_words=True):
    print(f'Loading {input_file}...')
    with open(input_file, encoding='utf-8') as f:
        if input_file.endswith('.conllu'):
            lines = [ [token['form'] for token in sentence] for sentence in conllu.parse(f.read())]
        else:
            lines = [ l.split() for l in f.readlines() if l.strip() != '' ]

        split_words = 0
        if token_format == 'char_to_word':
            all_lines = []
            for line in lines:
                words = []
                for word in line:
                    tokens = tokenizer.tokenize(word)
                    if split_long_words:
                        for i in range(0, len(tokens), max_word_length-1):
                            words.append(['[WORD_CLS]'] + tokens[i: i+max_word_length-1])
                    else:
                        words.append(['[WORD_CLS]'] + tokens[0: max_word_length-1])
                all_lines.append(words)
                split_words += len(words) - len(line)
        elif token_format == 'char':
            all_lines = [[['[WORD_CLS]', *tokenizer.tokenize(word)] for word in line] for line in lines]
        elif token_format == 'subword':
            all_lines = [[tokenizer.tokenize(word) for word in line] for line in lines]
        else:
            raise ValueError(f'Invalid token_format: {token_format}')

    # Save the encoded lines to the output file, splitting lines > max_seq_length
    total_lines = 0
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        if token_format == 'char_to_word':
            word_chunk_size = max_seq_length-2
            for line in all_lines:
                idx = 0
                while idx < len(line):
                    f.write(' '.join(token for word in line[idx:idx+word_chunk_size] for token in word) + '\n')
                    idx += word_chunk_size
                    total_lines += 1
        else:
            for line in all_lines:
                idx = 0
                char_chunk_size = max_seq_length-2
                # Split long lines at word boundaries
                while idx < len(line):
                    tokens = []
                    while idx < len(line) and len(tokens) + len(line[idx]) < char_chunk_size:
                        tokens.extend(line[idx])
                        idx += 1
                    f.write(' '.join(token for token in tokens) + '\n')
                    total_lines += 1

    split_lines = total_lines - len(lines)

    print(f'Saved {total_lines} lines to {output_file}')
    print(f'Split {split_lines} lines into multiple lines of max_seq_length={max_seq_length}')
    if split_words > 0:
        print(f'Split {split_words} words into multiple words of max_word_length={max_word_length}')

tokenizer = SPMTokenizer(args.vocab_path)
print(f"Train path: {train_path}")
encode(train_path, args.data_dir+"/train.txt", args.max_seq_length, args.max_word_length, args.token_format, tokenizer)
encode(valid_path, args.data_dir+"/valid.txt", args.max_seq_length, args.max_word_length, args.token_format, tokenizer)

train(asdict(args))
