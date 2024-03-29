from typing import List, Union
import sentencepiece as spm
from transformers import DebertaV2Tokenizer
import os
import shutil
import unicodedata
from conllu import parse

UPOS_TAGS = ('ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', '_')


def get_special_tag_tokens():
    upos_tokens = [f'[UPOS_{tag}]' for tag in UPOS_TAGS]
    return upos_tokens


def train_character_tokenizer(file_path: Union[str, os.PathLike], output_dir: Union[str, os.PathLike]):
    """ Train a character tokenizer that can be used by both DeBERTa and HuggingFace transformers. """
    with open(file_path, "r", encoding="utf-8") as file:
        data = parse(file.read())

    sents = [' '.join(token['form'] for token in sentence) for sentence in data]
    lemmas = [' '.join(token['lemma'] for token in sentence) for sentence in data]

    # SentencePiece performs unicode normalization internally, so we need to generate the character list from normalized text.
    train_sents = [unicodedata.normalize('NFKC', s) for s in sents + lemmas]

    os.makedirs(os.path.join(output_dir, 'spm'), exist_ok=True)
    model_prefix = os.path.join(output_dir, 'spm', 'spm')

    unique_chars = set(' '.join(train_sents))
    unique_chars.remove(' ')
    unique_chars.add('▁')
    unique_chars.discard('\ufeff')
    extra_symbols = ['[MASK]', '[WORD_CLS]', *get_special_tag_tokens()]
    vocab = unique_chars.union(extra_symbols).union(['[PAD]', '[UNK]', '[CLS]', '[SEP]'])

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(train_sents),
        model_prefix=model_prefix,
        vocab_size=len(vocab),
        character_coverage=1.0,
        add_dummy_prefix=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[CLS]',
        eos_piece='[SEP]',
        user_defined_symbols=','.join(extra_symbols),
        model_type='unigram',
    )

    # Convert to HuggingFace tokenizer so it can be loaded with AutoTokenizer.from_pretrained()
    tokenizer = DebertaV2Tokenizer(vocab_file=model_prefix + '.model')
    tokenizer.add_special_tokens({'additional_special_tokens': ['[WORD_CLS]']})
    tokenizer.save_pretrained(output_dir)

    # Don't need to keep the original file
    shutil.rmtree(os.path.join(output_dir, 'spm'))

    # Check that the tokenizer has the expected vocab
    loaded_vocab = set(tokenizer.get_vocab().keys())
    unexpected_tokens = loaded_vocab.difference(vocab)
    expected_tokens = vocab.difference(loaded_vocab)
    if len(unexpected_tokens) > 0:
        print('Warning: Unexpected tokens:', unexpected_tokens)
    if len(expected_tokens) > 0:
        print('Warning: Expected tokens:', expected_tokens)

    assert tokenizer.vocab_size == len(vocab)
    assert all(len(t) == 1 or (t[0] == '[' and t[-1] == ']') for t in tokenizer.get_vocab())

    return tokenizer


def train_subword_tokenizer(file_path: Union[str, os.PathLike], output_dir: Union[str, os.PathLike], vocab_size: int):
    """ Train a subword tokenizer that can be used by both DeBERTa and HuggingFace transformers. """
    with open(file_path, "r", encoding="utf-8") as file:
        data = parse(file.read())

    sents = [' '.join(token['form'] for token in sentence) for sentence in data]
    lemmas = [' '.join(token['lemma'] for token in sentence) for sentence in data]
    train_sents = sents + lemmas

    extra_symbols = ['[MASK]', *get_special_tag_tokens()]

    os.makedirs(os.path.join(output_dir, 'spm'), exist_ok=True)
    model_prefix = os.path.join(output_dir, 'spm', 'spm')
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(train_sents),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        add_dummy_prefix=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[CLS]',
        eos_piece='[SEP]',
        user_defined_symbols=','.join(extra_symbols),
        model_type='unigram',
    )

    # Convert to HuggingFace tokenizer so it can be loaded with AutoTokenizer.from_pretrained()
    tokenizer = DebertaV2Tokenizer(vocab_file=model_prefix + '.model')
    tokenizer.save_pretrained(output_dir)

    # Don't need to keep the original file
    shutil.rmtree(os.path.join(output_dir, 'spm'))

    return tokenizer

if __name__ == '__main__':
    languages = ['chu', 'cop', 'fro', 'got', 'grc', 'hbo', 'isl', 'lat', 'latm', 'lzh', 'ohu', 'orv', 'san']
    train_files = { language: f'data/ST2024/morphology/train/{language}_train.conllu' for language in languages }

    # use transliterated text for Classical Chinese
    #train_files['lzh'] = 'data/lzh_train_transliterated.conllu'

    print(f'Training character tokenizers for {len(languages)} languages: {languages}')
    tokenizers = []
    for language in languages:
        print('Training tokenizer for', language)
        tokenizer = train_character_tokenizer(train_files[language], f'tokenizers/{language}_char')
        tokenizers.append((language, tokenizer))

    for language, tokenizer in tokenizers:
        print(f'{language}: {tokenizer.vocab_size}')
