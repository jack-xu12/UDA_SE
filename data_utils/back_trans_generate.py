import torch
from transformers import MarianMTModel, MarianTokenizer
import argparse
import pandas as pd
import os
from datetime import datetime

# from tqdm import tqdm

'''
    use it in google colab
'''


parser = argparse.ArgumentParser(description='pytorch version replace for back translation')
# NOTICE: work_space path in project root dir
parser.add_argument('--train_file_path', type=str, default='./SOSC_test.csv')
parser.add_argument('--out_file_path', type=str, default='./bt_test.csv')
args = parser.parse_args()

if __name__ == '__main__':

    agrs = parser.parse_args()
    train_csv = args.train_file_path
    out_csv = args.out_file_path

    torch.cuda.empty_cache()

    en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr").cuda()

    fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en").cuda()

    df = pd.read_csv(train_csv, delimiter=",")
    ori_contents = df['content'].tolist()
    bt_ens = []

    for i, content in enumerate(ori_contents):
        content = [content]
        with torch.no_grad():
            start_time = datetime.now()
            # print('the en_fr starts')
            translated_tokens = en_fr_model.generate(
                **{k: v.cuda() for k, v in en_fr_tokenizer(content, return_tensors="pt", padding=True, truncation=True,
                                                           max_length=128).items()},
                do_sample=True,
                top_k=10,
                temperature=2.0,
            )
            in_fr = [en_fr_tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

            # end_time = datatime.now()
            # print('the en_fr spending time is {}'.format(end_time - start_time))

            # print('the fr_en starts')
            # start_time = datetime.now()
            bt_tokens = fr_en_model.generate(
                **{k: v.cuda() for k, v in
                   fr_en_tokenizer(in_fr, return_tensors="pt", padding=True, truncation=True, max_length=128).items()},
                do_sample=True,
                top_k=10,
                temperature=2.0,
            )
            bt_en = [fr_en_tokenizer.decode(t, skip_special_tokens=True) for t in bt_tokens]

            end_time = datetime.now()
            print('the {}th fr_en spending time is {}'.format(i, (end_time - start_time)))

            torch.cuda.empty_cache()
            bt_ens.append(bt_en[0])

    df_out = pd.DataFrame(columns=('content', 'label'))
    df_out['content'] = bt_ens
    df_out['label'] = 'unsup'

    df_out.to_csv(out_csv)