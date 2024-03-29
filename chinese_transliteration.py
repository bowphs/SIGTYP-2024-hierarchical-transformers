from conllu import parse

method = 'hanzipy' # 'pinyin', 'wubi', 'hanzipy'

sep_symbol = chr(ord('_')+50000)

if method == 'hanzipy':
    from hanzipy.decomposer import HanziDecomposer
    decomposer = HanziDecomposer()

    def to_chinese(text):
        pass

    def transliterate(text):
        output_string = ''
        for char in text:
            decomposition = decomposer.decompose(char)
            strokes = ''.join(decomposition['graphical']) + sep_symbol
            assert 'No glyph' not in strokes
            output_string += strokes
        return output_string

else:
    import pickle

    map_dict = None
    reverse_map_dict = None

    if method == 'pinyin':
        map_file = "transliteration_tables/chinese_to_pinyin.pkl"
        reverse_map_file = "transliteration_tables/pinyin_to_chinese.pkl"
    elif method == 'wubi':
        map_file = "transliteration_tables/chinese_to_wubi.pkl"
        reverse_map_file = "transliteration_tables/wubi_to_chinese.pkl"

    def to_chinese(text):
        global reverse_map_dict

        if reverse_map_dict is None:
            with open(reverse_map_file, 'rb') as f:
                reverse_map_dict = pickle.load(f)

        tokens = text.split(sep_symbol)
        output_string = ''
        for token in tokens:
            if token in reverse_map_dict:
                output_string += reverse_map_dict[token]
            else:
                assert (len(token) <= 1)
                output_string += token

        return output_string

    def transliterate(token):
        global map_dict

        if map_dict is None:
            with open(map_file, 'rb') as f:
                map_dict = pickle.load(f)

        CH2EN_PUNC = {f: t for f, t in zip(
                        u'，。！？【】（）％＃＠＆１２３４５６７８９０；：',
                        u',.!?[]()%#@&1234567890;:')}

        token = token.strip().lower()
        output_string = ""
        for char in token:
            assert (char not in CH2EN_PUNC)
            #char = CH2EN_PUNC.get(char, char)
            if char in map_dict:
                ## append transliterated char and separation symbol
                output_string += map_dict[char] + sep_symbol
            else:
                assert (not char.isalpha())
                #if char.isalpha():
                    #char = chr(ord(char)+50000)
                output_string += char
                
        return output_string

if __name__ == '__main__':
    files = [
        ('data/ST2024/morphology/train/lzh_train.conllu', 'data/lzh_train_transliterated.conllu'),
        ('data/ST2024/morphology/valid/lzh_valid.conllu', 'data/lzh_valid_transliterated.conllu'),
        #('data/ST2024/morphology/test/lzh_test.conllu', 'data/lzh_test_transliterated.conllu'),
    ]
    total_count = 0
    not_equal_count = 0
    lemmas = set()
    different_lemmas = set()
    words = set()
    word_lengths = []
    for input_file, output_file in files:
        with open(input_file, 'r') as f:
            data = f.read()
        sentences = parse(data)
        for sentence in sentences:
            for token in sentence:
                words.add(token['form'])
                lemmas.add(token['lemma'])
                if token['form'] != token['lemma']:
                    not_equal_count += 1
                    different_lemmas.add(token['lemma'])
                total_count += 1

                transliterated_form = transliterate(token['form'])
                transliterated_lemma = transliterate(token['lemma'])

                #assert (to_chinese(transliterated_form) == token['form'])
                #assert (to_chinese(transliterated_lemma) == token['lemma'])

                token['form'] = transliterated_form
                token['lemma'] = transliterated_lemma

                word_lengths.append(len(transliterated_form))
                word_lengths.append(len(transliterated_lemma))

        with open(output_file, 'w') as f:
            f.writelines(sentence.serialize() for sentence in sentences)
        print(f'{input_file} -> {output_file}')

    print('Longest word length:', max(word_lengths))
    print('# tokens (train+valid):', total_count)
    print('# unique words:', len(words))
    print('# unique lemmas:', len(lemmas))
    print('# words where lemma differs from the token form:', not_equal_count)
    print('# unique lemmas that differ from the token form:', len(different_lemmas))
