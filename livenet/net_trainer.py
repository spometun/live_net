import math
import typing

import numpy as np
import torch
import pandas as pd

from ai_libs.simple_log import LOG
from . import utils


def get_summary_stat(life_stat: pd.DataFrame):
    low_cut = life_stat[life_stat["type"] == "output_low_cut_ratio"]
    low_cut = low_cut["value"].mean()
    high_cut = life_stat[life_stat["type"] == "output_high_cut_ratio"]
    high_cut = high_cut["value"].mean()
    gradient = life_stat[life_stat["type"] == "gradient"]
    av_abs_grad = gradient["value"].abs().mean()
    parameter = life_stat[life_stat["type"] == "parameter"]
    max_abs_param = parameter["value"].abs().max()
    delta = life_stat[life_stat["type"] == "delta"]
    av_abs_delta = delta["value"].abs().mean()
    av_abs_output = life_stat[life_stat["type"] == "output_av"]
    av_abs_output = av_abs_output["value"].mean()
    max_abs_output = life_stat[life_stat["type"] == "output_max"]
    max_abs_output = max_abs_output["value"].mean()
    res = f"av_abs_grad {av_abs_grad:.1g} av_abs_delta: {av_abs_delta:.1g} av_abs_output {av_abs_output:.1g} max_abs_output {max_abs_output:.1g} max_abs_param {max_abs_param:.1f} hcut {high_cut:.1g} lcut {low_cut:.1f}"
    return res


class NetTrainer:
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
        self.adaptive_lr_increase_step = 1.005
        self.adaptive_lr_decrease_step = 2
        self.adaptive_lr_max_lr = 0.01
        self.adaptive_lr_min_lr = 1e-5
        self.last_epoch_tick = -1
        self.last_epoch_all_loss = math.inf
        self._n_loss_increases = 0
        self._need_to_stop = False
        self.clear_life_stat = True

    def step(self, n_steps=1):
        for _ in range(n_steps):
            if self._need_to_stop:
                LOG("stopped")
                break
            self._step()

    def _step(self):
        self.network.context.tick += 1
        data, labels = next(self.batch_iterator)
        device = next(self.network.parameters()).device
        data = torch.tensor(data, device=device)
        labels = torch.tensor(labels, device=device)
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
            pass
            # self._adjust_lr(data, labels, all_loss)

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
            if sign == "--":
                LOG(f"{sign} {old_lr:.5f} -> {new_lr:.5f}")
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
                             "loss": epoch_loss_criterion,
                             "loss_reg": epoch_loss_network})
        tick = self.network.context.tick
        msg = f"{tick}"
        msg += f" {epoch_loss_criterion:.3f}+{epoch_loss_network:.3f}reg"
        msg += f" params={len(params)}"
        if self.adaptive_lr:
            all_loss = epoch_loss_criterion + epoch_loss_network
            k = all_loss / self.last_epoch_all_loss
            if k >= 1:
                self._n_loss_increases += 1
            else:
                self._n_loss_increases = 0
            if self._n_loss_increases == 3:
                self.optimizer.learning_rate /= 2
            if self.optimizer.learning_rate < self.adaptive_lr_min_lr:
                self._need_to_stop = True
            msg += f" lr={self.optimizer.learning_rate:.5f}"
            self.last_epoch_all_loss = all_loss

        df = pd.DataFrame(self.network.context.life_stat)
        if len(df) > 0:
            df = df[df["tick"] > self.last_epoch_tick]
            msg += f" {get_summary_stat(df)}"
        if self.clear_life_stat:
            self.network.context.life_stat = []

        LOG(msg)
        self.loss_criterion = 0.0
        self.loss_network = 0.0
        self.counter_good = 0
        self.last_epoch_tick = tick


