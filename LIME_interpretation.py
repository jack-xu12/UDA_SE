import fire
from utils import optim, configuration
from utils.utils import get_device
from load_data import load_data
import models
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
import ast

from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from IPython.core.display import HTML, display
import pandas as pd
import csv

####### Inspect the model prediction with Lime #######
class LIME_class():

    def __init__(self, model, input_ids, segment_ids, input_mask, label_id):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.model = model


        self.input_mask_perturb = input_mask
        self.segment_ids_perturb = segment_ids  # we don't change this variable ever

        lasso_lime_base = LimeBase(
            self.forward_func,
            interpretable_model=SkLearnLasso(alpha=0.08),
            similarity_func=self.exp_embedding_cosine_distance,
            perturb_func=self.bernoulli_perturb,
            perturb_interpretable_space=True,
            from_interp_rep_transform=self.interp_to_input,
            to_interp_rep_transform=None
        )

        self.attrs = lasso_lime_base.attribute(
            self.input_ids,  # add batch dimension for Captum
            target=self.label_id.squeeze(),
            n_samples=40000,
            show_progress=True
        ).squeeze(0)


    # remove the batch dimension for the embedding-bag model
    def forward_func(self, input_ids):
        return self.model(input_ids, self.segment_ids_perturb, self.input_mask_perturb)

    # encode text indices into latent representations & calculate cosine similarity
    def exp_embedding_cosine_distance(self, original_inp, perturbed_inp, _, **kwargs):
        def get_embedding(inp, segment_ids, input_mask):
            h = self.model.transformer(inp, segment_ids, input_mask)
            # only use the first h in the sequence
            pooled_h = self.model.activ(self.model.fc(h[:, 0]))
            return pooled_h

        original_emb = get_embedding(original_inp, self.segment_ids, self.input_mask)
        perturbed_emb = get_embedding(perturbed_inp, self.segment_ids_perturb, self.input_mask_perturb)
        distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=1)
        return torch.exp(-1 * (distance ** 2) / 2)

    # binary vector where each word is selected independently and uniformly at random
    def bernoulli_perturb(self, text, **kwargs):

        _end_idx = torch.sum(self.input_mask).item()
        probs = self.input_mask * 0.5
        probs[0, 0] = 1
        probs[0, _end_idx-1] = 1
        return torch.bernoulli(probs).long()

    # remove absenst token based on the intepretable representation sample
    def interp_to_input(self, interp_sample, original_input, **kwargs):

        # in the bert embedding model, to change this, we just need to change mask vector
        assert interp_sample.size() == self.input_mask.size()
        _perturb = interp_sample.bool() & self.input_mask.bool()
        self.input_mask_perturb = _perturb.type_as(self.input_mask)

        return  original_input

        # return original_input[interp_sample.bool()].view(original_input.size(0), -1)


def show_text_attr(attrs, idx, token_file):

    ####### load token file #######
    df = pd.read_csv(token_file, encoding='utf-8')
    _idx_tokens = ast.literal_eval(df.loc[idx, 'tokens'])

    rgb = lambda x: '255,0,0' if x < 0 else '0,255,0'
    alpha = lambda x: abs(x) ** 0.5
    token_marks = [
        f'<mark style="background-color:rgba({rgb(attr)},{alpha(attr)})">{token}</mark>'
        for token, attr in zip(_idx_tokens, attrs.tolist())
    ]

    html_str = '<p>' + ' '.join(token_marks) + '</p>'

    return html_str



def write_csv(path, writing_mode, data_row):

    with open(path, writing_mode, encoding='utf-8') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)




def main(cfg, model_cfg):


    # selected_idxs = [88, 90, 188, 196]

    # Load Configuration
    cfg = configuration.params.from_json(cfg)  # Train or Eval cfg
    model_cfg = configuration.model.from_json(model_cfg)  # BERT_cfg
    device = get_device()

    data = load_data(cfg)
    assert cfg.mode == 'eval', "## mode in cfg must be 'eval' ## "
    data_iter = data.sup_data_iter()

    # Load Model
    model = models.Classifier(model_cfg, len(data.TaskDataset.labels))

    token_file = 'preprocessed_data/sosc_data/back/sosc_sup_test_last_1_tokens.csv'
    out_csv = 'LIME_interpretation_html.csv'
    # load model from pretrain file
    model_file = cfg.model_file
    model.eval()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_file))
    model = model.to(device)
    if cfg.data_parallel:
        raise RuntimeError('we just need [1, dim] batch forward, no need parallel here')
        # model = nn.DataParallel(model)
    iter_bar = tqdm(deepcopy(data_iter))

    html_strs = []
    attr_strs = []

    # write the head
    write_csv(out_csv, writing_mode='w', data_row=['attr_strs', 'html_strs'])

    for idx, batch in enumerate(iter_bar):
        # if not idx in selected_idxs:
        #     continue
        _b = [t.to(device) for t in batch]
        input_ids, segment_ids, input_mask, label_id = _b
        _lime_interpret = LIME_class(model = model,
                   input_ids= input_ids,
                   segment_ids= segment_ids,
                   input_mask= input_mask,
                   label_id = label_id,)

        attrs = _lime_interpret.attrs
        print('Attribution range:', attrs.min().item(), 'to', attrs.max().item())
        _attrs = ','.join([str(a) for a in attrs.numpy().tolist()])
        html_str = show_text_attr(attrs, idx, token_file=token_file)

        attr_strs.append(_attrs)
        html_strs.append(html_str)

        write_csv(out_csv, writing_mode='a+', data_row=[_attrs, html_str])




    # data_dict = {"html_strs": html_strs}
    # pd.DataFrame(data_dict).to_csv('LIME_interpretation_html.csv', index=False, encoding='utf-8')




if __name__ == '__main__':
    fire.Fire(main)


