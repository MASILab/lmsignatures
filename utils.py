import os
import random
import torch
import numpy as np
import yaml

HEAD = "linear_head"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_config(config_dir, config_id):
    with open(os.path.join(config_dir, f"{config_id}.yaml")) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

class EarlyStopper():
    def __init__(self, agg=200, delta=0.005):
        """
        Stopping criteria: running median over agg steps is worse than any previous median by more than delta
        :param agg: number of steps to aggregate metric over
        :param delta: maximum change in running median before stopping
        self.current_step counts the number of val performed, not the global steps
        """
        self.history = []
        self.medians = []
        self.agg = agg
        self.delta = delta
        self.current_step = 0
    
    def step(self, v):
        self.history.append(v)
        self.current_step += 1
    
    def loss_check_stop(self):
        # stop if current median HIGHER than previous median by delta
        if self.current_step < self.agg:
            return False
        else:
            # running median in agg range
            current = np.median(self.history[self.current_step-self.agg:self.current_step])
            self.medians.append(current)
            # check if current median worse
            for m in self.medians:
                if current > (m + self.delta):
                    return True
            return False
        
    def acc_check_stop(self):
        # stop if current median LOWER than previous median by delta
        if self.current_step < self.agg:
            return False
        else:
            # running median in agg range
            current = np.median(self.history[self.current_step-self.agg:self.current_step])
            self.medians.append(current)
            # check if current median worse 
            for m in self.medians:
                if current < (m - self.delta):
                    return True
            return False
            
"""
Schedulers
https://github.com/jeonsworld/ViT-pytorch/utils/scheduler.py
"""
import logging
import math
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))