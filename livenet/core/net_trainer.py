import typing

import numpy as np
import torch
from simple_log import LOG
from . import utils


class Trainer:
    def __init__(self, network: torch.nn.Module, batch_iterator: typing.Iterator,
                 criterion: typing.Callable, optimizer, epoch_size=1, adaptive_lr=False):
        self.network = network
        self.batch_iterator = batch_iterator
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch_size = epoch_size
        self.history = []
        self.counter = 0
        self.counter_good = 0
        self.loss_criterion = 0.0
        self.loss_network = 0.0
        self.adaptive_lr = adaptive_lr
        self.adaptive_lr_increase_step = 1.02
        self.adaptive_lr_decrease_step = 1.2
        self.adaptive_lr_max_lr = 0.1
        self.adaptive_lr_min_lr = 0.00001

    def step(self, n_steps=1):
        for _ in range(n_steps):
            self._step()

    def _step(self):
        data, labels = next(self.batch_iterator)
        pred = self.network.forward(data)

        loss = self.criterion(pred, labels)
        self.loss_criterion += loss.detach().item()
        loss_network = self.network.internal_loss()
        self.loss_network += loss_network.detach().item()
        all_loss = loss + loss_network

        self.optimizer.zero_grad()
        all_loss.backward()
        self.optimizer.step()

        if self.adaptive_lr:
            self._adjust_lr(data, labels, all_loss)

        self.counter += 1
        if self.counter % self.epoch_size == 0:
            self._on_epoch()

    def _adjust_lr(self, data, labels, all_loss):
        with torch.no_grad():
            pred1 = self.network.forward(data)
            loss1 = self.criterion(pred1, labels)
            loss1_network = self.network.internal_loss()
            all_loss1 = loss1 + loss1_network
            is_good = all_loss1.detach().item() < all_loss.detach().item()
            self.counter_good += is_good

            old_lr = self.optimizer.learning_rate
            if is_good:
                new_lr = old_lr * self.adaptive_lr_increase_step
                sign = "+ "
            else:
                new_lr = old_lr / self.adaptive_lr_decrease_step
                sign = "--"
            new_lr = np.clip(new_lr, self.adaptive_lr_min_lr, self.adaptive_lr_max_lr)
            # LOG(f"{sign} {old_lr:.5f} -> {new_lr:.5f}")
            self.optimizer.learning_rate = new_lr

    def _on_epoch(self):
        params = utils.get_parameters_dict(self.network)
        grads = utils.get_gradients_dict(self.network)
        epoch_loss_criterion = self.loss_criterion
        epoch_loss_network = self.loss_network
        good_ratio = self.counter_good / self.epoch_size
        # if self.counter != 0:
        epoch_loss_criterion /= self.epoch_size
        epoch_loss_network /= self.epoch_size
        self.history.append({"params": params,
                             "grads": grads,
                             "loss": epoch_loss_criterion,
                             "loss_reg": epoch_loss_network})
        msg = f"{epoch_loss_criterion + epoch_loss_network:.3f}"
        # if epoch_loss_network != 0.0:
        msg += f" = {epoch_loss_criterion:.3f}+{epoch_loss_network:.3f}"
        msg += f" params={len(params)}"
        if self.adaptive_lr:
            msg += f" lr={self.optimizer.learning_rate:.4f}"
        LOG(msg)
        self.loss_criterion = 0.0
        self.loss_network = 0.0
        self.counter_good = 0

