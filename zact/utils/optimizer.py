import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, config, current_step):

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            betas=config["betas"],
            eps=config["eps"],
            weight_decay=config["weight_decay"],
        )
        self.n_warmup_steps = config["warm_up_step"]
        self.anneal_steps = config["anneal_steps"]
        self.anneal_rate = config["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(config["encoder_hidden"], -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self.optimizer.zero_grad()

    def load_state_dict(self, path):
        self.optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class NoamScheduler(_LRScheduler):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, factor, model_size, warmup_steps):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        self.factor = factor
        self._rate = 0
    
    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict) 
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))) 