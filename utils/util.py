import os, sys, pdb
import math
import torch
from PIL import Image
import numpy as np
import random
import torch
import numpy as np
import os
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def gen_A(num_classes, adj_file, t=0.4, sig=0.3):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    # _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    # _adj = _adj + np.identity(num_classes, np.int)
    _adj = (1-sig)*_adj + (sig / (_adj.sum(0, keepdims=True) + 1e-6))
    _adj = _adj + np.identity(num_classes, np.int64)
    return _adj

def get_vec(vec_file):
    import pickle
    word_vec = pickle.load(open(vec_file, 'rb'))
    return word_vec['vector']

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    sd = model.state_dict()
    for key, value in checkpoint['model'].items():
        if key != 'head.weight' and key != 'head.bias':     # TODO should comment this line when continuing from checkpoint (not pretrain)
            sd[key] = value
    model.load_state_dict(sd)
    # checkpoint['model']['head.weight'] = torch.zeros(2, model.num_features)
    # checkpoint['model']['head.bias'] = torch.zeros(2)
    # msg = model.load_state_dict(checkpoint['model'], strict=False)
    # logger.info(msg)
    logger.info("Pretrain model loaded successfully!")
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        #config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt



def multiAuc(y_true, y_pred):
    auc = []

    for i in range(y_true.shape[1]):
        auc.append(roc_auc_score(y_true=y_true[:, i], y_score=y_pred[:, i]))

    return auc


def plotRoc(y_true, y_score, is_valid):
    auc = multiAuc(y_true=y_true, y_pred=y_score)
    label = ['D', 'G', 'C', 'A', 'H', 'M', 'O']
    lw = 2
    _, c = y_true.shape
    plt.figure()

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    for i in range(c):
        fpr, tpr, _ = roc_curve(y_true=y_true[:, i], y_score=y_score[:, i])
        plt.plot(fpr, tpr, lw=lw, label=f'AUC of {label[i]}:{auc[i]:.2f}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC Socre Mean:{np.mean(auc):.2f}')
    plt.legend(loc='lower right')
    if is_valid:
        plt.savefig('validroc.png')
    else:
        plt.savefig('roc.png')
    
    
