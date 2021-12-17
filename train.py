# Copyright 2019 SanghunYun, Korea University.
# (Strongly inspired by Dong-Hyun Lee, Kakao Brain)
# 
# Except load and save function, the whole codes of file has been modified and added by
# SanghunYun, Korea University for UDA.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from copy import deepcopy
from typing import NamedTuple
from tqdm import tqdm

import torch
import torch.nn as nn

from utils import checkpoint
# from utils.logger import Logger
from tensorboardX import SummaryWriter
from utils.utils import output_logging
from sklearn.metrics import precision_recall_fscore_support


class Trainer(object):
    """Training Helper class"""
    def __init__(self, cfg, model, data_iter, optimizer, device, save_pt):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_pt = save_pt

        # preprocessed_data iter
        if len(data_iter) == 1:
            # this is just for the eval mode, sup_iter -- eval_iter
            self.sup_iter = data_iter[0]
        elif len(data_iter) == 2:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
        elif len(data_iter) == 3:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            if data_iter[1] is not None:
                self.unsup_iter = self.repeat_dataloader(data_iter[1])
            self.eval_iter = data_iter[2]

    def train(self, get_loss, get_acc, model_file, pretrain_file):
        """ train uda"""

        # tensorboardX logging
        if self.cfg.results_dir:
            logger = SummaryWriter(log_dir=os.path.join(self.cfg.results_dir, 'logs'))


        self.model.train()
        self.load(model_file, pretrain_file)    # between model_file and pretrain_file, only one model will be loaded
        model = self.model.to(self.device)
        if self.cfg.data_parallel:                       # Parallel GPU mode
            model = nn.DataParallel(model)

        global_step = 0
        loss_sum = 0.
        max_acc = [0., 0]   # acc, step

        max_f1_c0 = [0, 0, 0]  # p_score r_score f1_score for class 0
        max_f1_c1 = [0, 0, 0]  # p_score r_score f1_score for class 1
        max_f1_c2 = [0, 0, 0]  # p_score r_score f1_score for class 2

        if self.cfg.data_parallel:
            n_labels = model.module.classifier.out_features
        else:
            n_labels = model.classifier.out_features

        max_f1 = [max_f1_c0, max_f1_c1]
        if n_labels == 3:
            max_f1 = [max_f1_c0, max_f1_c1, max_f1_c2]


        # Progress bar is set by unsup or sup preprocessed_data
        # uda_mode == True --> sup_iter is repeated
        # uda_mode == False --> sup_iter is not repeated
        iter_bar = tqdm(self.unsup_iter, total=self.cfg.total_steps) if self.cfg.uda_mode \
              else tqdm(self.sup_iter, total=self.cfg.total_steps)
        for i, batch in enumerate(iter_bar):

                
            # Device assignment
            if self.cfg.uda_mode:
                sup_batch = [t.to(self.device) for t in next(self.sup_iter)]
                unsup_batch = [t.to(self.device) for t in batch]
            else:
                sup_batch = [t.to(self.device) for t in batch]
                unsup_batch = None

            # update
            self.optimizer.zero_grad()
            final_loss, sup_loss, unsup_loss = get_loss(model, sup_batch, unsup_batch, global_step)
            final_loss.backward()
            self.optimizer.step()

            # print loss
            global_step += 1
            loss_sum += final_loss.item()
            if self.cfg.uda_mode:
                iter_bar.set_description('final=%5.3f unsup=%5.3f sup=%5.3f'\
                        % (final_loss.item(), unsup_loss.item(), sup_loss.item()))
            else:
                iter_bar.set_description('loss=%5.3f' % (final_loss.item()))

            # logging            
            if self.cfg.uda_mode:
                logger.add_scalars('preprocessed_data/scalar_group',
                                    {'final_loss': final_loss.item(),
                                     'sup_loss': sup_loss.item(),
                                     'unsup_loss': unsup_loss.item(),
                                     'lr': self.optimizer.get_lr()[0]
                                    }, global_step)
            else:
                logger.add_scalars('preprocessed_data/scalar_group',
                                    {'sup_loss': final_loss.item()}, global_step)

            # if global_step % self.cfg.save_steps == 0:
            #     self.save(global_step)

            if get_acc and global_step % self.cfg.check_steps == 0 and global_step > 499:
                max_acc, max_f1 = self._update_metrics(get_acc, model, logger, global_step,
                                     max_acc, max_f1)

            if self.cfg.total_steps and self.cfg.total_steps < global_step:
                print('The total steps have been reached')
                print('Average Loss %5.3f' % (loss_sum/(i+1)))
                if get_acc:
                    max_acc, max_f1 = self._update_metrics(get_acc, model, logger, global_step,
                                         max_acc, max_f1)

                if len(max_f1) == 3:

                    log_str = f'{self.cfg.sup_data_dir}\n' \
                              f'max_acc: {max_acc}\n' \
                              f'neg: p:{max_f1[0][0]} r:{max_f1[0][1]} f1:{max_f1[0][2]}\n' \
                              f'neu: p:{max_f1[1][0]} r:{max_f1[1][1]} f1:{max_f1[1][2]}\n' \
                              f'pos: p:{max_f1[2][0]} r:{max_f1[2][1]} f1:{max_f1[2][2]}\n'
                else:

                    log_str = f'{self.cfg.sup_data_dir}\n' \
                              f'max_acc: {max_acc}\n' \
                              f'pos: p:{max_f1[0][0]} r:{max_f1[0][1]} f1:{max_f1[0][2]}\n' \
                              f'neg: p:{max_f1[1][0]} r:{max_f1[1][1]} f1:{max_f1[1][2]}\n'


                with open(self.cfg.log_file, 'a+', encoding='utf-8') as f:
                    f.write(log_str)

                return
        return global_step

    def _update_metrics(self, get_acc, model, logger, global_step, max_acc, max_f1):

        n_labels = len(max_f1)
        if n_labels == 2:
            max_f1_c0, max_f1_c1 = max_f1
        else:
            max_f1_c0, max_f1_c1, max_f1_c2 = max_f1

        results, preds, labels = self.eval(get_acc, None, model)

        _preds = torch.cat(preds).detach().cpu().numpy()
        _labels = torch.cat(labels).detach().cpu().numpy()

        p_class, r_class, f_class, _ = precision_recall_fscore_support(_labels, _preds)

        p_c0, r_c0, f_c0 = p_class[0], r_class[0], f_class[0]
        p_c1, r_c1, f_c1 = p_class[1], r_class[1], f_class[1]
        if n_labels == 3:
            p_c2, r_c2, f_c2 = p_class[2], r_class[2], f_class[2]

        logger.add_scalar('f_score_c0/p', p_c0, global_step)
        logger.add_scalar('f_score_c0/r', r_c0, global_step)
        logger.add_scalar('f_score_c0/f', f_c0, global_step)

        if max_f1_c0[-1] < f_c0:
            max_f1_c0 = [p_c0, r_c0, f_c0]

        logger.add_scalar('f_score_c1/p', p_c1, global_step)
        logger.add_scalar('f_score_c1/r', r_c1, global_step)
        logger.add_scalar('f_score_c1/f', f_c1, global_step)

        if max_f1_c1[-1] < f_c1:
            max_f1_c1 = [p_c1, r_c1, f_c1]

        max_f1 = [max_f1_c0, max_f1_c1]
        if n_labels == 3:

            logger.add_scalar('f_score_c2/p', p_c2, global_step)
            logger.add_scalar('f_score_c2/r', r_c2, global_step)
            logger.add_scalar('f_score_c2/f', f_c2, global_step)

            if max_f1_c2[-1] < f_c2:
                max_f1_c2 = [p_c2, r_c2, f_c2]
            max_f1 = [max_f1_c0, max_f1_c1, max_f1_c2]

        import numpy as np
        total_accuracy = np.sum(_preds==_labels) / _preds.shape[-1]
        logger.add_scalars('preprocessed_data/scalar_group', {'eval_acc': total_accuracy}, global_step)
        if max_acc[0] < total_accuracy:
            if self.save_pt:
                self.save(global_step)
            max_acc = total_accuracy, global_step
        print('Accuracy : %5.3f' % total_accuracy)
        print(
            'Max Accuracy : %5.3f Max global_steps : %d Cur global_steps : %d' % (max_acc[0], max_acc[1], global_step),
            end='\n\n')

        return max_acc, max_f1

    def eval(self, evaluate, model_file, model):
        """ evaluation function """
        if model_file:
            self.model.eval()
            self.load(model_file, None)
            model = self.model.to(self.device)
            if self.cfg.data_parallel:
                model = nn.DataParallel(model)

        results = []
        preds = []
        labels = []

        iter_bar = tqdm(self.sup_iter) if model_file \
            else tqdm(deepcopy(self.eval_iter))
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]

            with torch.no_grad():
                accuracy, result, y_pred, y_label = evaluate(model, batch)
            results.append(result)
            preds.append(y_pred)
            labels.append(y_label)

            iter_bar.set_description('Eval Acc=%5.3f' % accuracy)
        return results, preds, labels
            
    def load(self, model_file, pretrain_file):
        """ between model_file and pretrain_file, only one model will be loaded """
        if model_file:
            print('Loading the model from', model_file)
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(model_file))
            else:
                self.model.load_state_dict(torch.load(model_file, map_location='cpu'))

        elif pretrain_file:
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'):  # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'):  # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                )   # load only transformer parts
    
    def save(self, i):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg.results_dir, 'save')):
            os.makedirs(os.path.join(self.cfg.results_dir, 'save'))
        torch.save(self.model.state_dict(),
                        os.path.join(self.cfg.results_dir, 'save', 'model_steps_'+str(i)+'.pt'))

    def repeat_dataloader(self, iterable):
        """ repeat dataloader """
        while True:
            for x in iterable:
                yield x
