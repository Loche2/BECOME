from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from model import *
from .utils import *
from .loggers import *
from .dataset import *
from .dataloader import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes

import json
import faiss
import numpy as np
from abc import *
from pathlib import Path

import copy



class AutoDataRankDistillationTrainer(metaclass=ABCMeta):
    def __init__(self, args, model_code, model, bb_model, test_loader, export_root, loss='ranking', tau=1., margin_topk=0.5, margin_neg=1.0):
        self.args = args
        self.device = args.device
        self.num_items = args.num_items
        if args.auto_budget:
            self.max_len = args.bert_max_len
        else:
            self.max_len = args.bert_max_len * args.pass_top_percent
            self.max_len = int(self.max_len)
        print('max_len:', self.max_len)
        self.batch_size = args.train_batch_size
        self.mask_prob = args.bert_mask_prob
        self.max_predictions = args.bert_max_predictions
        self.CLOZE_MASK_TOKEN = self.num_items + 1

        self.model = model.to(self.device)
        self.model_code = model_code
        self.bb_model = bb_model.to(self.device)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.export_root = export_root
        self.log_period_as_iter = args.log_period_as_iter

        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            if args.enable_lr_warmup:
                self.lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.optimizer, args.warmup_steps, (args.num_generated_seqs // self.batch_size + 1) * self.num_epochs * 2)
            else:
                self.lr_scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.loss = loss
        self.tau = tau
        self.margin_topk = margin_topk
        self.margin_neg = margin_neg
        if self.loss == 'kl':
            self.loss_func = nn.KLDivLoss(reduction='batchmean')
        elif self.loss == 'ranking':
            self.loss_func_1 = nn.MarginRankingLoss(margin=self.margin_topk)
            self.loss_func_2 = nn.MarginRankingLoss(margin=self.margin_neg)
        elif self.loss == 'kl+ct':
            self.loss_func_1 = nn.KLDivLoss(reduction='batchmean')
            self.loss_func_2 = nn.CrossEntropyLoss(ignore_index=0)

        # AutoBudget Args
        self.auto_budget = args.auto_budget
        self.pass_top_percent = args.pass_top_percent
        self.auto_round_num = args.auto_round_num
        self.auto_round_epoch = args.auto_round_epoch
        self.selected_items_count = torch.zeros(self.num_items + 1).to(self.device)

        # Active Learning Args
        self.active_learning = args.active_learning

    def calculate_loss(self, seqs, labels, candidates, lengths=None):
        if isinstance(self.model, BERT) or isinstance(self.model, SASRec):
            logits = self.model(seqs)[:, -1, :]
        elif isinstance(self.model, NARM):
            logits = self.model(seqs, lengths)
        
        if self.loss == 'kl':
            logits = torch.gather(logits, -1, candidates)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1, labels.size(-1))
            loss = self.loss_func(F.log_softmax(logits/self.tau, dim=-1), F.softmax(labels/self.tau, dim=-1))
        
        elif self.loss == 'ranking':
            # logits = F.softmax(logits/self.tau, dim=-1)
            weight = torch.ones_like(logits).to(self.device)
            weight[torch.arange(weight.size(0)).unsqueeze(1), candidates] = 0
            neg_samples = torch.distributions.Categorical(F.softmax(weight, -1)).sample_n(candidates.size(-1)*30).permute(1, 0)
            # assume candidates are in descending order w.r.t. true label
            neg_logits = torch.gather(logits, -1, neg_samples)

            # _, sorted_logits = torch.topk(logits, k=100, dim=-1, largest=True, sorted=True)
            
            logits = torch.gather(logits, -1, candidates)
            logits_1 = logits[:, :-1].reshape(-1)
            logits_2 = logits[:, 1:].reshape(-1)
            loss = self.loss_func_1(logits_1, logits_2, torch.ones(logits_1.shape).to(self.device))
            loss += self.loss_func_2(logits.repeat_interleave(30, dim=-1), neg_logits, torch.ones(logits.repeat_interleave(30, dim=-1).shape).to(self.device))
            # loss += self.loss_func_2(logits, neg_logits, torch.ones(logits.shape).to(self.device))


            # error_list = generate_error_lists(sorted_logits, candidates)
            # logits_Vl = torch.gather(logits, -1, torch.tensor(error_list['Vl'], dtype=torch.long).to(logits.device))
            # logits_Vh = torch.gather(logits, -1, torch.tensor(error_list['Vh'], dtype=torch.long).to(logits.device))
            # logits_Tl = torch.gather(logits, -1, torch.tensor(error_list['Tl'], dtype=torch.long).to(logits.device))
            # logits_Th = torch.gather(logits, -1, torch.tensor(error_list['Th'], dtype=torch.long).to(logits.device))

            # repari_loss_low = self.loss_func_1(logits_Vl, logits_Tl, torch.ones(logits_Vl.shape).to(self.device))
            # repari_loss_high = self.loss_func_1(logits_Vh, logits_Th, torch.ones(logits_Vh.shape).to(self.device))

            # loss += repari_loss_low + repari_loss_high
            
        elif self.loss == 'kl+ct':
            logits = torch.gather(logits, -1, candidates)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1, labels.size(-1))
            loss = self.loss_func_1(F.log_softmax(logits/self.tau, dim=-1), F.softmax(labels/self.tau, dim=-1))
            loss += self.loss_func_2(F.softmax(logits), torch.argmax(labels, dim=-1))
        return loss

    def calculate_metrics(self, batch, similarity=False):
        self.model.eval()
        self.bb_model.eval()

        if isinstance(self.model, BERT) or isinstance(self.model, SASRec):
            seqs, candidates, labels = batch
            seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
            scores = self.model(seqs)[:, -1, :]
            metrics = recalls_and_ndcgs_for_ks(scores.gather(1, candidates), labels, self.metric_ks)
        elif isinstance(self.model, NARM):
            seqs, lengths, candidates, labels = batch
            seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
            lengths = lengths.flatten()
            scores = self.model(seqs, lengths)
            metrics = recalls_and_ndcgs_for_ks(scores.gather(1, candidates), labels, self.metric_ks)

        if similarity:
            if isinstance(self.model, BERT) and isinstance(self.bb_model, BERT):
                soft_labels = self.bb_model(seqs)[:, -1, :]
            elif isinstance(self.model, BERT) and isinstance(self.bb_model, SASRec):
                temp_seqs = torch.cat((torch.zeros(seqs.size(0)).long().unsqueeze(1).to(self.device), seqs[:, :-1]), dim=1)
                soft_labels = self.bb_model(temp_seqs)[:, -1, :]
            elif isinstance(self.model, BERT) and isinstance(self.bb_model, NARM):
                temp_seqs = torch.cat((torch.zeros(seqs.size(0)).long().unsqueeze(1).to(self.device), seqs[:, :-1]), dim=1)
                temp_seqs = self.pre2post_padding(temp_seqs)
                temp_lengths = (temp_seqs > 0).sum(-1).cpu().flatten()
                soft_labels = self.bb_model(temp_seqs, temp_lengths)
            elif isinstance(self.model, SASRec) and isinstance(self.bb_model, SASRec):
                soft_labels = self.bb_model(seqs)[:, -1, :]
            elif isinstance(self.model, SASRec) and isinstance(self.bb_model, BERT):
                temp_seqs = torch.cat((seqs[:, 1:], torch.tensor([self.CLOZE_MASK_TOKEN] * seqs.size(0)).unsqueeze(1).to(self.device)), dim=1)
                soft_labels = self.bb_model(temp_seqs)[:, -1, :]
            elif isinstance(self.model, SASRec) and isinstance(self.bb_model, NARM):
                temp_seqs = self.pre2post_padding(seqs)
                temp_lengths = (temp_seqs > 0).sum(-1).cpu().flatten()
                soft_labels = self.bb_model(temp_seqs, temp_lengths)
            elif isinstance(self.model, NARM) and isinstance(self.bb_model, NARM):
                soft_labels = self.bb_model(seqs, lengths)
            elif isinstance(self.model, NARM) and isinstance(self.bb_model, BERT):
                temp_seqs = self.post2pre_padding(seqs)
                temp_seqs = torch.cat((temp_seqs[:, 1:], torch.tensor([self.CLOZE_MASK_TOKEN] * seqs.size(0)).unsqueeze(1).to(self.device)), dim=1)
                soft_labels = self.bb_model(temp_seqs)[:, -1, :]
            elif isinstance(self.model, NARM) and isinstance(self.bb_model, SASRec):
                temp_seqs = self.post2pre_padding(seqs)
                soft_labels = self.bb_model(temp_seqs)[:, -1, :]
            similarity, similarity_each = kl_agreements_and_intersctions_for_ks(scores, soft_labels, self.metric_ks, auto_budget=self.auto_budget)
            metrics = {**metrics, **similarity} 
        
        return metrics, similarity_each

    def generate_autoregressive_data(self, auto_budget=False, auto_round=0, k=100, batch_size=50, agreements_each=None, auto_round_len=20):
        dataset = dis_dataset_factory(self.args, self.model_code, 'autoregressive')
        # if dataset.check_data_present():
        #     print('Dataset already exists. Skip generation')
        #     return
        
        batch_num = self.args.num_generated_seqs // batch_size
        print('Generating dataset...')

        if auto_budget:
            if auto_round == 0:
                self.max_len = auto_round_len
                for i in tqdm(range(batch_num)):
                    seqs = torch.randint(1, self.num_items + 1, (batch_size, 1)).to(self.device)
                    logits = None
                    candidates = None
                    
                    self.bb_model.eval()
                    with torch.no_grad():
                        if isinstance(self.bb_model, BERT):
                            mask_items = torch.tensor([self.CLOZE_MASK_TOKEN] * seqs.size(0)).to(self.device)
                            for j in range(self.max_len - 1):
                                input_seqs = torch.zeros((seqs.size(0), self.max_len)).to(self.device)
                                input_seqs[:, (self.max_len-2-j):-1] = seqs
                                input_seqs[:, -1] = mask_items
                                labels = self.bb_model(input_seqs.long())[:, -1, :]

                                _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
                                sorted_items = sorted_items[:, :k] + 1
                                randomized_label = torch.rand(sorted_items.shape).to(self.device)
                                randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                                randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)

                                if self.active_learning:
                                    selected_indices = self.active_learning_selection(randomized_label, sorted_items, input_seqs)
                                else:
                                    selected_indices = torch.distributions.Categorical(F.softmax(torch.ones_like(randomized_label), -1).to(randomized_label.device)).sample()

                                row_indices = torch.arange(sorted_items.size(0))
                                seqs = torch.cat((seqs, sorted_items[row_indices, selected_indices].unsqueeze(1)), 1)
                                # seqs = torch.cat((seqs, sorted_items[row_indices, 0].unsqueeze(1)), 1)

                                try:
                                    logits = torch.cat((logits, randomized_label.unsqueeze(1)), 1)
                                    candidates = torch.cat((candidates, sorted_items.unsqueeze(1)), 1)
                                except:
                                    logits = randomized_label.unsqueeze(1)
                                    candidates = sorted_items.unsqueeze(1)

                            input_seqs = torch.zeros((seqs.size(0), self.max_len)).to(self.device)
                            input_seqs[:, :-1] = seqs[:, 1:]
                            input_seqs[:, -1] = mask_items
                            labels = self.bb_model(input_seqs.long())[:, -1, :]
                            _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
                            sorted_items = sorted_items[:, :k] + 1
                            randomized_label = torch.rand(sorted_items.shape).to(self.device)
                            randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                            randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)
                            
                            logits = torch.cat((logits, randomized_label.unsqueeze(1)), 1)
                            candidates = torch.cat((candidates, sorted_items.unsqueeze(1)), 1)

                        elif isinstance(self.bb_model, SASRec):
                            for j in range(self.max_len - 1):
                                input_seqs = torch.zeros((seqs.size(0), self.max_len)).to(self.device)
                                input_seqs[:, (self.max_len-1-j):] = seqs
                                labels = self.bb_model(input_seqs.long())[:, -1, :]

                                _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                                sorted_items = sorted_items[:, :k] + 1
                                randomized_label = torch.rand(sorted_items.shape).to(self.device)
                                randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                                randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)
                            
                                if self.active_learning:
                                    selected_indices = self.active_learning_selection(randomized_label, sorted_items, input_seqs)
                                else:
                                    selected_indices = torch.distributions.Categorical(F.softmax(torch.ones_like(randomized_label), -1).to(randomized_label.device)).sample()
                                
                                row_indices = torch.arange(sorted_items.size(0))
                                seqs = torch.cat((seqs, sorted_items[row_indices, selected_indices].unsqueeze(1)), 1)

                                try:
                                    logits = torch.cat((logits, randomized_label.unsqueeze(1)), 1)
                                    candidates = torch.cat((candidates, sorted_items.unsqueeze(1)), 1)
                                except:
                                    logits = randomized_label.unsqueeze(1)
                                    candidates = sorted_items.unsqueeze(1)

                            labels = self.bb_model(seqs.long())[:, -1, :]
                            _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                            sorted_items = sorted_items[:, :k] + 1
                            randomized_label = torch.rand(sorted_items.shape).to(self.device)
                            randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                            randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)
                            
                            logits = torch.cat((logits, randomized_label.unsqueeze(1)), 1)
                            candidates = torch.cat((candidates, sorted_items.unsqueeze(1)), 1)

                        elif isinstance(self.bb_model, NARM):
                            for j in range(self.max_len - 1):
                                lengths = torch.tensor([j + 1] * seqs.size(0))
                                labels = self.bb_model(seqs.long(), lengths)

                                _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                                sorted_items = sorted_items[:, :k] + 1
                                randomized_label = torch.rand(sorted_items.shape).to(self.device)
                                randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                                randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True) 

                                if self.active_learning:
                                    selected_indices = self.active_learning_selection(randomized_label, sorted_items, seqs)
                                else:
                                    selected_indices = torch.distributions.Categorical(F.softmax(torch.ones_like(randomized_label), -1).to(randomized_label.device)).sample()

                                row_indices = torch.arange(sorted_items.size(0))
                                seqs = torch.cat((seqs, sorted_items[row_indices, selected_indices].unsqueeze(1)), 1)
                                
                                try:
                                    logits = torch.cat((logits, randomized_label.unsqueeze(1)), 1)
                                    candidates = torch.cat((candidates, sorted_items.unsqueeze(1)), 1)
                                except:
                                    logits = randomized_label.unsqueeze(1)
                                    candidates = sorted_items.unsqueeze(1)

                            lengths = torch.tensor([self.max_len] * seqs.size(0))
                            labels = self.bb_model(seqs.long(), lengths)
                            _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                            sorted_items = sorted_items[:, :k] + 1
                            randomized_label = torch.rand(sorted_items.shape).to(self.device)
                            randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                            randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)
                            
                            logits = torch.cat((logits, randomized_label.unsqueeze(1)), 1)
                            candidates = torch.cat((candidates, sorted_items.unsqueeze(1)), 1)

                        if i == 0:
                            batch_tokens = seqs.cpu().numpy()
                            batch_logits = logits.cpu().numpy()
                            batch_candidates = candidates.cpu().numpy()
                        else:
                            batch_tokens = np.concatenate((batch_tokens, seqs.cpu().numpy()))
                            batch_logits = np.concatenate((batch_logits, logits.cpu().numpy()))
                            batch_candidates = np.concatenate((batch_candidates, candidates.cpu().numpy()))

                dataset.save_dataset(batch_tokens.tolist(), batch_logits.tolist(), batch_candidates.tolist())
            else:
                self.max_len += auto_round_len
                dataset_temp = dataset.load_dataset()
                for i in tqdm(range(batch_num)):
                    seqs = dataset_temp['seqs'][i * batch_size: (i + 1) * batch_size]
                    logits = dataset_temp['logits'][i * batch_size: (i + 1) * batch_size]
                    candidates = dataset_temp['candidates'][i * batch_size: (i + 1) * batch_size]
                    # print('size:', len(seqs[0]), len(logits[0]), len(candidates[0]))
                    
                    # 添加序列选择器功能
                    # 可以基于某些条件来选择需要更新的序列
                    seq_indices_to_update = self.filter_sequences_for_update(seqs, auto_round, agreements_each[i * batch_size: (i + 1) * batch_size])
                    selected_seqs = [seqs[i] for i in seq_indices_to_update]
                    selected_logits = None
                    selected_candidates = None
                    
                    if isinstance(self.bb_model, BERT) or isinstance(self.bb_model, SASRec):
                        selected_seqs = [[0] * (self.max_len - auto_round_len - len(seq)) + seq for seq in selected_seqs]
                    elif isinstance(self.bb_model, NARM):
                        selected_seqs = [seq + [0] * (self.max_len - auto_round_len - len(seq)) for seq in selected_seqs]
                    else:
                        print('Model not supported???')
                    selected_seqs = torch.tensor(selected_seqs).to(self.device)

                    self.bb_model.eval()
                    with torch.no_grad():
                        if isinstance(self.bb_model, BERT):
                            mask_items = torch.tensor([self.CLOZE_MASK_TOKEN] * selected_seqs.size(0)).to(self.device)
                            for j in range(self.max_len - auto_round_len -1, self.max_len - 1):
                                input_seqs = torch.zeros((selected_seqs.size(0), self.max_len)).to(self.device)
                                
                                input_seqs[:, (self.max_len-2-j):-1] = selected_seqs
                                input_seqs[:, -1] = mask_items
                                labels = self.bb_model(input_seqs.long())[:, -1, :]

                                _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
                                sorted_items = sorted_items[:, :k] + 1
                                randomized_label = torch.rand(sorted_items.shape).to(self.device)
                                randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                                randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)

                                if self.active_learning:
                                    selected_indices = self.active_learning_selection(randomized_label, sorted_items, input_seqs)
                                else:
                                    selected_indices = torch.distributions.Categorical(F.softmax(torch.ones_like(randomized_label), -1).to(randomized_label.device)).sample()
                                row_indices = torch.arange(sorted_items.size(0))
                                selected_seqs = torch.cat((selected_seqs, sorted_items[row_indices, selected_indices].unsqueeze(1)), 1)
                                # selected_seqs = torch.cat((selected_seqs, sorted_items[row_indices, 0].unsqueeze(1)), 1)

                                try:
                                    selected_logits = torch.cat((selected_logits, randomized_label.unsqueeze(1)), 1)
                                    selected_candidates = torch.cat((selected_candidates, sorted_items.unsqueeze(1)), 1)
                                except:
                                    selected_logits = randomized_label.unsqueeze(1)
                                    selected_candidates = sorted_items.unsqueeze(1)
                                
                            input_seqs = torch.zeros((selected_seqs.size(0), self.max_len)).to(self.device)
                            input_seqs[:, :-1] = selected_seqs[:,1:]
                            input_seqs[:, -1] = mask_items
                            labels = self.bb_model(input_seqs.long())[:, -1, :]

                            _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
                            sorted_items = sorted_items[:, :k] + 1
                            randomized_label = torch.rand(sorted_items.shape).to(self.device)
                            randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                            randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)
                            
                            selected_logits = torch.cat((selected_logits, randomized_label.unsqueeze(1)), 1)
                            selected_candidates = torch.cat((selected_candidates, sorted_items.unsqueeze(1)), 1)

                        elif isinstance(self.bb_model, SASRec):
                            for j in range(self.max_len - auto_round_len -1, self.max_len - 1):
                                input_seqs = torch.zeros((selected_seqs.size(0), self.max_len)).to(self.device)
                                input_seqs[:, (self.max_len-1-j):] = selected_seqs
                                labels = self.bb_model(input_seqs.long())[:, -1, :]

                                _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                                sorted_items = sorted_items[:, :k] + 1
                                randomized_label = torch.rand(sorted_items.shape).to(self.device)
                                randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                                randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)
                                
                                if self.active_learning:
                                    selected_indices = self.active_learning_selection(randomized_label, sorted_items, input_seqs)
                                else:
                                    selected_indices = torch.distributions.Categorical(F.softmax(torch.ones_like(randomized_label), -1).to(randomized_label.device)).sample()
                                
                                row_indices = torch.arange(sorted_items.size(0))
                                selected_seqs = torch.cat((selected_seqs, sorted_items[row_indices, selected_indices].unsqueeze(1)), 1)

                                try:
                                    selected_logits = torch.cat((selected_logits, randomized_label.unsqueeze(1)), 1)
                                    selected_candidates = torch.cat((selected_candidates, sorted_items.unsqueeze(1)), 1)
                                except:
                                    selected_logits = randomized_label.unsqueeze(1)
                                    selected_candidates = sorted_items.unsqueeze(1)

                            labels = self.bb_model(selected_seqs.long())[:, -1, :]
                            _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                            sorted_items = sorted_items[:, :k] + 1
                            randomized_label = torch.rand(sorted_items.shape).to(self.device)
                            randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                            randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)
                            
                            selected_logits = torch.cat((selected_logits, randomized_label.unsqueeze(1)), 1)
                            selected_candidates = torch.cat((selected_candidates, sorted_items.unsqueeze(1)), 1)

                        elif isinstance(self.bb_model, NARM):
                            # selected_seqs = [seq + [0] * (self.max_len - auto_round_len - len(seq)) for seq in selected_seqs]
                            # print(selected_seqs[0])
                            # selected_seqs = torch.tensor(selected_seqs).to(self.device)
                            # print(selected_seqs[0].size())
                            for j in range(self.max_len - auto_round_len - 1, self.max_len - 1):
                                lengths = torch.tensor([j + 1] * selected_seqs.size(0))
                                labels = self.bb_model(selected_seqs.long(), lengths)

                                _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                                sorted_items = sorted_items[:, :k] + 1
                                randomized_label = torch.rand(sorted_items.shape).to(self.device)
                                randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                                randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True) 

                                if self.active_learning:
                                    selected_indices = self.active_learning_selection(randomized_label, sorted_items, selected_seqs)
                                else:
                                    selected_indices = torch.distributions.Categorical(F.softmax(torch.ones_like(randomized_label), -1).to(randomized_label.device)).sample()
                                row_indices = torch.arange(sorted_items.size(0))
                                selected_seqs = torch.cat((selected_seqs, sorted_items[row_indices, selected_indices].unsqueeze(1)), 1)
                                
                                try:
                                    selected_logits = torch.cat((selected_logits, randomized_label.unsqueeze(1)), 1)
                                    selected_candidates = torch.cat((selected_candidates, sorted_items.unsqueeze(1)), 1)
                                except:
                                    selected_logits = randomized_label.unsqueeze(1)
                                    selected_candidates = sorted_items.unsqueeze(1)

                            lengths = torch.tensor([self.max_len] * selected_seqs.size(0))
                            labels = self.bb_model(selected_seqs.long(), lengths)
                            _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                            sorted_items = sorted_items[:, :k] + 1
                            randomized_label = torch.rand(sorted_items.shape).to(self.device)
                            randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                            randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)
                            
                            selected_logits = torch.cat((selected_logits, randomized_label.unsqueeze(1)), 1)
                            selected_candidates = torch.cat((selected_candidates, sorted_items.unsqueeze(1)), 1)

                        updated_seqs = copy.deepcopy(seqs)
                        updated_logits = copy.deepcopy(logits)
                        updated_candidates = copy.deepcopy(candidates)

                        selected_seqs = [seq[seq != 0] for seq in selected_seqs]
                        # 更新选定的序列
                        for n, idx in enumerate(seq_indices_to_update):
                            updated_seqs[idx] = selected_seqs[n].cpu().tolist()
                            # if updated_logits is not None and selected_logits is not None:
                            # if not isinstance(self.bb_model, NARM):
                            updated_logits[idx].pop()
                            updated_logits[idx].extend(selected_logits[n].cpu().tolist())
                            # if updated_candidates is not None and selected_candidates is not None:
                            # if not isinstance(self.bb_model, NARM):
                            updated_candidates[idx].pop()
                            updated_candidates[idx].extend(selected_candidates[n].cpu().tolist())
                        if i == 0:
                            batch_tokens = updated_seqs
                            batch_logits = updated_logits
                            batch_candidates = updated_candidates
                        else:
                            batch_tokens.extend(updated_seqs)
                            batch_logits.extend(updated_logits)
                            batch_candidates.extend(updated_candidates)
                
                    print('batch_tokens:',len(batch_tokens[0]))
                    print('batch_logits:', len(batch_logits[0]))
                    print('batch_candidates:', len(batch_candidates[0]))
                
                # 保存更新后的数据集
                dataset.save_dataset(batch_tokens, batch_logits, batch_candidates)

        else:
            for i in tqdm(range(batch_num)):
                seqs = torch.randint(1, self.num_items + 1, (batch_size, 1)).to(self.device)
                logits = None
                candidates = None
                
                self.bb_model.eval()
                with torch.no_grad():
                    if isinstance(self.bb_model, BERT):
                        mask_items = torch.tensor([self.CLOZE_MASK_TOKEN] * seqs.size(0)).to(self.device)
                        for j in range(self.max_len - 1):
                            input_seqs = torch.zeros((seqs.size(0), self.max_len)).to(self.device)
                            input_seqs[:, (self.max_len-2-j):-1] = seqs
                            input_seqs[:, -1] = mask_items
                            labels = self.bb_model(input_seqs.long())[:, -1, :]

                            _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
                            sorted_items = sorted_items[:, :k] + 1
                            randomized_label = torch.rand(sorted_items.shape).to(self.device)
                            randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                            randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)

                            if self.active_learning:
                                selected_indices = self.active_learning_selection(randomized_label, sorted_items, input_seqs)
                            else:
                                selected_indices = torch.distributions.Categorical(F.softmax(torch.ones_like(randomized_label), -1).to(randomized_label.device)).sample()
                            row_indices = torch.arange(sorted_items.size(0))
                            seqs = torch.cat((seqs, sorted_items[row_indices, selected_indices].unsqueeze(1)), 1)
                            # # 随机从项目池中选择一个
                            # random_indices = torch.randint(0, sorted_items.size(1), (seqs.size(0),), device=seqs.device)
                            # seqs = torch.cat((seqs, sorted_items[row_indices, random_indices].unsqueeze(1)), 1)
                            # seqs = torch.cat((seqs, sorted_items[row_indices, 0].unsqueeze(1)), 1)

                            try:
                                logits = torch.cat((logits, randomized_label.unsqueeze(1)), 1)
                                candidates = torch.cat((candidates, sorted_items.unsqueeze(1)), 1)
                            except:
                                logits = randomized_label.unsqueeze(1)
                                candidates = sorted_items.unsqueeze(1)

                        input_seqs = torch.zeros((seqs.size(0), self.max_len)).to(self.device)
                        input_seqs[:, :-1] = seqs[:, 1:]
                        input_seqs[:, -1] = mask_items
                        labels = self.bb_model(input_seqs.long())[:, -1, :]
                        _, sorted_items = torch.sort(labels[:, 1:-1], dim=-1, descending=True)
                        sorted_items = sorted_items[:, :k] + 1
                        randomized_label = torch.rand(sorted_items.shape).to(self.device)
                        randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                        randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)
                        
                        logits = torch.cat((logits, randomized_label.unsqueeze(1)), 1)
                        candidates = torch.cat((candidates, sorted_items.unsqueeze(1)), 1)

                    elif isinstance(self.bb_model, SASRec):
                        for j in range(self.max_len - 1):
                            input_seqs = torch.zeros((seqs.size(0), self.max_len)).to(self.device)
                            input_seqs[:, (self.max_len-1-j):] = seqs
                            labels = self.bb_model(input_seqs.long())[:, -1, :]

                            _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                            sorted_items = sorted_items[:, :k] + 1
                            randomized_label = torch.rand(sorted_items.shape).to(self.device)
                            randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                            randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)
                            
                            if self.active_learning:
                                selected_indices = self.active_learning_selection(randomized_label, sorted_items, input_seqs)
                            else:
                                selected_indices = torch.distributions.Categorical(F.softmax(torch.ones_like(randomized_label), -1).to(randomized_label.device)).sample()
                            row_indices = torch.arange(sorted_items.size(0))
                            seqs = torch.cat((seqs, sorted_items[row_indices, selected_indices].unsqueeze(1)), 1)

                            try:
                                logits = torch.cat((logits, randomized_label.unsqueeze(1)), 1)
                                candidates = torch.cat((candidates, sorted_items.unsqueeze(1)), 1)
                            except:
                                logits = randomized_label.unsqueeze(1)
                                candidates = sorted_items.unsqueeze(1)

                        labels = self.bb_model(seqs.long())[:, -1, :]
                        _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                        sorted_items = sorted_items[:, :k] + 1
                        randomized_label = torch.rand(sorted_items.shape).to(self.device)
                        randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                        randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)
                        
                        logits = torch.cat((logits, randomized_label.unsqueeze(1)), 1)
                        candidates = torch.cat((candidates, sorted_items.unsqueeze(1)), 1)

                    elif isinstance(self.bb_model, NARM):
                        for j in range(self.max_len - 1):
                            lengths = torch.tensor([j + 1] * seqs.size(0))
                            labels = self.bb_model(seqs.long(), lengths)

                            _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                            sorted_items = sorted_items[:, :k] + 1
                            randomized_label = torch.rand(sorted_items.shape).to(self.device)
                            randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                            randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True) 

                            if self.active_learning:
                                selected_indices = self.active_learning_selection(randomized_label, sorted_items, seqs)
                            else:
                                selected_indices = torch.distributions.Categorical(F.softmax(torch.ones_like(randomized_label), -1).to(randomized_label.device)).sample()
                            row_indices = torch.arange(sorted_items.size(0))
                            seqs = torch.cat((seqs, sorted_items[row_indices, selected_indices].unsqueeze(1)), 1)
                            
                            try:
                                logits = torch.cat((logits, randomized_label.unsqueeze(1)), 1)
                                candidates = torch.cat((candidates, sorted_items.unsqueeze(1)), 1)
                            except:
                                logits = randomized_label.unsqueeze(1)
                                candidates = sorted_items.unsqueeze(1)

                        lengths = torch.tensor([self.max_len] * seqs.size(0))
                        labels = self.bb_model(seqs.long(), lengths)
                        _, sorted_items = torch.sort(labels[:, 1:], dim=-1, descending=True)
                        sorted_items = sorted_items[:, :k] + 1
                        randomized_label = torch.rand(sorted_items.shape).to(self.device)
                        randomized_label = randomized_label / randomized_label.sum(dim=-1).unsqueeze(-1)
                        randomized_label, _ = torch.sort(randomized_label, dim=-1, descending=True)
                        
                        logits = torch.cat((logits, randomized_label.unsqueeze(1)), 1)
                        candidates = torch.cat((candidates, sorted_items.unsqueeze(1)), 1)

                    if i == 0:
                        batch_tokens = seqs.cpu().numpy()
                        batch_logits = logits.cpu().numpy()
                        batch_candidates = candidates.cpu().numpy()

                    else:
                        batch_tokens = np.concatenate((batch_tokens, seqs.cpu().numpy()))
                        batch_logits = np.concatenate((batch_logits, logits.cpu().numpy()))
                        batch_candidates = np.concatenate((batch_candidates, candidates.cpu().numpy()))

            dataset.save_dataset(batch_tokens.tolist(), batch_logits.tolist(), batch_candidates.tolist())

    def train_autoregressive(self):
        if self.auto_budget:
            accum_iter = 0
            self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
            self.logger_service = LoggerService(
                self.train_loggers, self.val_loggers)
            print('## Distilling model via autoregressive data... ##')
            agreements_each = None
            auto_round_len_ = self.max_len // (self.auto_round_num + 1)
            random_len_ = random_len(self.max_len-auto_round_len_, self.auto_round_num-1)
            for round in range(self.auto_round_num):
                print('## AutoBudget Round {}: ##'.format(round + 1))
                if round != 0:
                    auto_round_len_ = random_len_[round-1]
                    print(random_len_)
                    print(auto_round_len_)
                self.generate_autoregressive_data(self.auto_budget, auto_round=round, agreements_each=agreements_each, auto_round_len=auto_round_len_)
                dis_train_loader, dis_val_loader = dis_train_loader_factory(self.args, self.model_code, 'autoregressive')
                
                if round == 8:
                    self.auto_round_epoch = 30
                for epoch in range(self.auto_round_epoch):
                    accum_iter = self.train_one_epoch(epoch, accum_iter, dis_train_loader, dis_val_loader, stage=round + 1)

                agreements_each = self.validate(dis_val_loader, 0, accum_iter)
                print(agreements_each)
                print(agreements_each.mean())

            torch.save(self.selected_items_count, 'selected_items_count.pt')

            metrics = self.test()
        
            self.logger_service.complete({
                'state_dict': (self._create_state_dict()),
            })
            self.writer.close()
            return metrics
            
        else:
            accum_iter = 0
            self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
            self.logger_service = LoggerService(
                self.train_loggers, self.val_loggers)
            self.generate_autoregressive_data()
            dis_train_loader, dis_val_loader = dis_train_loader_factory(self.args, self.model_code, 'autoregressive')
            print('## Distilling model via autoregressive data... ##')
            self.validate(dis_val_loader, 0, accum_iter)
            for epoch in range(self.num_epochs):
                accum_iter = self.train_one_epoch(epoch, accum_iter, dis_train_loader, dis_val_loader, stage=1)
            
            metrics = self.test()
            
            self.logger_service.complete({
                'state_dict': (self._create_state_dict()),
            })
            self.writer.close()
            return metrics

    def train_one_epoch(self, epoch, accum_iter, train_loader, val_loader, stage=0):
        self.model.train()
        self.bb_model.train()
        average_meter_set = AverageMeterSet()
        
        tqdm_dataloader = tqdm(train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            self.optimizer.zero_grad()
            if isinstance(self.model, BERT) or isinstance(self.model, SASRec):
                seqs, candidates, labels = batch
                seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                loss = self.calculate_loss(seqs, labels, candidates)
            elif isinstance(self.model, NARM):
                seqs, lengths, candidates, labels = batch
                lengths = lengths.flatten()
                seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                loss = self.calculate_loss(seqs, labels, candidates, lengths=lengths)
            
            loss.backward()
            self.clip_gradients(5)
            self.optimizer.step()
            accum_iter += int(seqs.size(0))
            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {} Stage {}, loss {:.3f} '.format(epoch+1, stage, average_meter_set['loss'].avg))

            if self._needs_to_log(accum_iter):
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch+1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.logger_service.log_train(log_data)
            
            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()
        
        self.validate(val_loader, epoch, accum_iter)
        return accum_iter

    def validate(self, val_loader, epoch, accum_iter):
        agr_each = np.empty((0,))
        self.model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                metrics, metrics_each = self.calculate_metrics(batch, similarity=True)
                # print('Agr@10:', metrics['Agr@10'])
                # print('Agr_each@10:', type(metrics_each['Agr_each@10']))
                agr_each = np.concatenate((agr_each, metrics_each['Agr_each@10']))
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_val(log_data)

        return agr_each

    def test(self):
        wb_model = torch.load(os.path.join(
            self.export_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
        self.model.load_state_dict(wb_model)

        self.model.eval()
        self.bb_model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                metrics, _ = self.calculate_metrics(batch, similarity=True)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)
                
            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
        return average_metrics
    
    def filter_sequences_for_update(self, seqs, round, agreements):
        """
        根据agreements_each选择需要更新的序列，不更新前10%高agreement的序列
        
        参数:
        - seqs: 所有序列张量
        - round: 当前轮次
        
        返回:
        - 布尔张量，表示需要更新的序列索引
        """
        # print('len(agreements):', len(agreements))
        # 初始化为全部更新
        num_seqs = len(seqs)
        
        # 如果是第一轮或没有agreements，则更新所有序列
        # if round == 0 or agreements is None:
        #     print(f"Round {round}: Updating all sequences")
        #     # 返回所有索引
        #     return list(range(num_seqs))
        
        # 如果agreements的长度与序列数不匹配，则更新所有序列
        if len(agreements) != num_seqs:
            print(f"Warning: agreements length {len(agreements)} != seqs length {num_seqs}, updating all sequences")
            return list(range(num_seqs))
        
        # 将agreements转为张量并按降序排序获取索引
        agreements_tensor = torch.tensor(agreements)
        _, sorted_indices = torch.sort(agreements_tensor, descending=True)
        
        # 计算前10%的cutoff索引
        top_percent = self.pass_top_percent # 前10%
        cutoff_idx = int(num_seqs * top_percent)
        
        # 获取前10%高agreement的序列索引 - 这些不更新
        top_indices = sorted_indices[:cutoff_idx].cpu().numpy()
        
        # 创建要更新的序列索引列表 (排除top索引)
        update_indices = [i for i in range(num_seqs) if i not in top_indices]
        # update_indices = [i for i in range(num_seqs)]
        
        print(f"Round {round + 1}: Not updating {cutoff_idx} sequences with highest agreement values")
        print(update_indices)
        
        return update_indices
    
    def active_learning_selection(self, randomized_label, sorted_items, input_seqs):
        # 计算选择每个候选项的多样性分数
        diversity_scores = 1.0 / (torch.log(self.selected_items_count[sorted_items] + 1.0) + 1.0)

        alpha = 0.8
        sampling_prob = F.softmax(
            alpha * F.softmax(diversity_scores, dim=-1) + (1 - alpha) * F.softmax(randomized_label, dim=-1),
            dim=-1
        )
        selected_indices = torch.distributions.Categorical(sampling_prob).sample()

        # 向量化更新已选择的项目计数
        row_indices = torch.arange(sorted_items.size(0), device=sorted_items.device)
        selected_indices = selected_indices.to(dtype=torch.long, device=sorted_items.device)
        selected_items = sorted_items[row_indices, selected_indices]
        self.selected_items_count.index_add_(
            0, selected_items, torch.ones_like(selected_items, dtype=self.selected_items_count.dtype)
        )

        return selected_indices


    def bb_model_test(self):
        self.bb_model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                if isinstance(self.model, BERT) or isinstance(self.model, SASRec):
                    seqs, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    scores = self.bb_model(seqs)[:, -1, :]
                    metrics = recalls_and_ndcgs_for_ks(scores.gather(1, candidates), labels, self.metric_ks)
                elif isinstance(self.model, NARM):
                    seqs, lengths, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    lengths = lengths.flatten()
                    scores = self.bb_model(seqs, lengths)
                    metrics = recalls_and_ndcgs_for_ks(scores.gather(1, candidates), labels, self.metric_ks)

                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
        
        return average_metrics

    def pre2post_padding(self, seqs):
        processed = torch.zeros_like(seqs)
        lengths = (seqs > 0).sum(-1).squeeze()
        for i in range(seqs.size(0)):
            processed[i, :lengths[i]] = seqs[i, seqs.size(1)-lengths[i]:]
        return processed

    def post2pre_padding(self, seqs):
        processed = torch.zeros_like(seqs)
        lengths = (seqs > 0).sum(-1).squeeze()
        for i in range(seqs.size(0)):
            processed[i, seqs.size(1)-lengths[i]:] = seqs[i, :lengths[i]]
        return processed

    def clip_gradients(self, limit=5):
        for p in self.model.parameters():
            nn.utils.clip_grad_norm_(p, 5)

    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(self, tqdm_dataloader, meter_set):
        description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]
                               ] + ['Recall@%d' % k for k in self.metric_ks[:3]]
        description = 'Eval: ' + \
            ', '.join(s + ' {:.3f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        description = description.format(
            *(meter_set[k].avg for k in description_metrics))
        tqdm_dataloader.set_description(description)

    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        # based on hugging face get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch',
                               graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss',
                               graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(
            model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
