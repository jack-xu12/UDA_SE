
import collections
import six
import pandas as pd
import ast



def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    idx_vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()   # sanghun 특수문자 제거  문장만 추출
            vocab[token] = index    # index 부여
            idx_vocab[index] = token
            index += 1


    return vocab, idx_vocab

if __name__ == '__main__':

    vocab_file = 'BERT_Base_Uncased/vocab.txt'
    eval_file = 'preprocessed_data/sosc_data/back/sosc_sup_test_last_1.csv'

    f = open(eval_file, 'r', encoding='utf-8')
    data = pd.read_csv(f, sep=',')

    input_columns = ['input_ids', 'input_type_ids', 'input_mask', 'label_ids']
    datas = data[input_columns[0]].apply(lambda x: ast.literal_eval(x))

    vocab, idx_vocab = load_vocab(vocab_file)
    tokens = []
    for d in datas:
        _tokens = [idx_vocab[idx] for idx in d]
        tokens.append(_tokens)

    data_dict = {'tokens': tokens}
    df_new = pd.DataFrame(data_dict)

    df_new.to_csv('preprocessed_data/sosc_data/back/sosc_sup_test_last_1_tokens.csv', index=False, encoding='utf-8')




