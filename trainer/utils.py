from config import *

import json
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim as optim


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                         for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}
    # metrics_each = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)

    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(
                labels.device), labels.sum(1).float())).mean().cpu().item()
        
        # metrics_each['Recall_each@%d' % k] = (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).cpu()

        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                             for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()

        # metrics_each['NDCG_each@%d' % k] = (dcg / idcg).cpu()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()
    return metrics


def em_and_agreement(scores_rank, labels_rank):
    em = (scores_rank == labels_rank).float().mean()
    temp = np.hstack((scores_rank.numpy(), labels_rank.numpy()))
    temp = np.sort(temp, axis=1)
    agreement = np.mean(np.sum(temp[:, 1:] == temp[:, :-1], axis=1))
    agreement_each = np.sum(temp[:, 1:] == temp[:, :-1], axis=1)
    return em, agreement, agreement_each


def kl_agreements_and_intersctions_for_ks(scores, soft_labels, ks, k_kl=100, auto_budget=False):
    metrics = {}
    metrics_each = {}
    scores = scores.cpu()
    soft_labels = soft_labels.cpu()
    scores_rank = (-scores).argsort(dim=1)
    labels_rank = (-soft_labels).argsort(dim=1)

    top_kl_scores = F.log_softmax(scores.gather(1, labels_rank[:, :k_kl]), dim=-1)
    top_kl_labels = F.softmax(soft_labels.gather(1, labels_rank[:, :k_kl]), dim=-1)
    kl = F.kl_div(top_kl_scores, top_kl_labels, reduction='batchmean')
    metrics['KL-Div'] = kl.item()
    for k in sorted(ks, reverse=True):
        em, agreement, agreement_each = em_and_agreement(scores_rank[:, :k], labels_rank[:, :k])
        metrics['EM@%d' % k] = em.item()
        metrics['Agr@%d' % k] = (agreement / k).item()
        metrics_each['Agr_each@%d' % k] = agreement_each / k
    return metrics, metrics_each


def lambda_loss(scores: torch.Tensor, relevance: torch.Tensor):
    """
    支持 batch 的基于 NDCG 的 LambdaLoss 实现，shape = [B, N]
    scores: 模型预测打分, shape = [B, N]
    relevance: 真实相关性标签, shape = [B, N]
    """
    B, N = scores.shape
    device = scores.device

    loss = 0.0
    for b in range(B):
        s = scores[b]       # [N]
        r = relevance[b]    # [N]

        # 获取模型排序位置
        _, rank_indices = torch.sort(s, descending=True)
        ranks = torch.zeros_like(s, dtype=torch.float)
        ranks[rank_indices] = torch.arange(N, dtype=torch.float, device=device)

        # 构造 pairwise
        i_idx = torch.arange(N, device=device).unsqueeze(1).repeat(1, N).view(-1)
        j_idx = torch.arange(N, device=device).repeat(N)

        rel_i = r[i_idx]
        rel_j = r[j_idx]
        score_i = s[i_idx]
        score_j = s[j_idx]
        rank_i = ranks[i_idx]
        rank_j = ranks[j_idx]

        # 仅保留 rel_i > rel_j 的正负对
        valid_pair = rel_i > rel_j
        if valid_pair.sum() == 0:
            continue  # 跳过无有效对的样本

        delta_ndcg = torch.abs(1.0 / torch.log2(rank_i[valid_pair] + 2.0) -
                               1.0 / torch.log2(rank_j[valid_pair] + 2.0))

        pair_loss = F.softplus(-(score_i[valid_pair] - score_j[valid_pair]))
        sample_loss = (delta_ndcg * pair_loss).sum() / delta_ndcg.sum().clamp(min=1e-6)
        loss += sample_loss

    return loss / B


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

def generate_error_lists(blackbox_list, whitebox_list):
    """
    生成Vl, Vh, Tl, Th四个列表的完整逻辑
    :param blackbox_list: 黑盒推荐列表（按排名顺序排列的item_id）
    :param whitebox_list: 白盒推荐列表（按预测排名顺序排列的item_id）
    :return: Vl, Vh, Tl, Th四个列表的字典
    """
    # 预处理数据结构
    black_rank = {item: idx+1 for idx, item in enumerate(blackbox_list)}  # 黑盒项目排名（从1开始）
    white_rank = {item: idx+1 for idx, item in enumerate(whitebox_list)}  # 白盒项目排名（从1开始）
    black_set = set(blackbox_list)  # 用于快速存在性检查
    
    # 第一阶段：生成Vl和Vh
    Vl, Vh = [], []
    for item in blackbox_list:
        if item not in white_rank: continue  # 白盒未推荐则跳过
        if black_rank[item] > white_rank[item]:  # 黑盒排名更靠后（低估）
            Vl.append(item)
        elif black_rank[item] < white_rank[item]:  # 黑盒排名更靠前（过估）
            Vh.append(item)

    # 第二阶段：生成Tl和Th
    def find_target_items(ref_rank, search_range, condition):
        targets = []
        for item in search_range:
            pos = ref_rank - 1  # 转换为0-based索引
            if pos >= len(whitebox_list): continue
            # 按条件查找目标位置
            while 0 <= pos < len(whitebox_list):
                candidate = whitebox_list[pos]
                if candidate not in black_set:
                    targets.append(candidate)
                    break  # 找到第一个符合条件的即停止
                pos += 1 if condition == 'later' else -1
        return list(dict.fromkeys(targets))  # 去重保持顺序

    # 生成Tl（基于Vl的黑盒排名位置向后查找）
    Tl = find_target_items(
        ref_rank=np.mean([black_rank[item] for item in Vl]) if Vl else 0,
        search_range=Vl,
        condition='later'
    )

    # 生成Th（基于Vh的黑盒排名位置向前查找）
    Th = find_target_items(
        ref_rank=np.mean([black_rank[item] for item in Vh]) if Vh else 0,
        search_range=Vh,
        condition='earlier'
    )

    return {'Vl': Vl, 'Vh': Vh, 'Tl': Tl, 'Th': Th}

def random_len(target_sum, num_elements, mean=0, std_dev=1):
    """
    生成一个指定元素个数且元素和为指定值的整数列表，元素通过正态分布随机生成。

    :param target_sum: 列表的目标和
    :param num_elements: 列表的元素个数
    :param mean: 正态分布的均值
    :param std_dev: 正态分布的标准差
    :return: 生成的整数列表
    """
    if num_elements <= 0:
        raise ValueError("元素个数必须大于 0")
    if target_sum < num_elements:
        raise ValueError("目标和必须大于或等于元素个数（每个元素至少为 1）")

    # 随机生成 num_elements 个正态分布的数
    random_numbers = np.random.normal(mean, std_dev, num_elements)
    # 将随机数调整为正数并归一化
    positive_numbers = np.abs(random_numbers)
    normalized_numbers = positive_numbers / sum(positive_numbers) * target_sum
    # 将归一化后的数四舍五入为整数
    integer_numbers = np.round(normalized_numbers).astype(int)

    # 调整整数列表的和为 target_sum
    diff = target_sum - sum(integer_numbers)
    for i in range(abs(diff)):
        index = i % num_elements
        if diff > 0:
            integer_numbers[index] += 1
        elif diff < 0 and integer_numbers[index] > 0:
            integer_numbers[index] -= 1

    return integer_numbers.tolist()