__package__ = "livenet"
import torch

from .core.livenet import Context, RegularNeuron, DestinationNeuron, SourceNeuron, DataNeuron
from .core import optimizers, livenet
from . import nets, gen_utils, net_trainer
from . import datasets
from simple_log import LOG


def test_die():
    context = Context(seed=42)
    context.module = torch.nn.Module()
    src = DataNeuron(context)
    src.set_output(torch.Tensor(42))
    neuron = RegularNeuron(context, activation=torch.nn.ReLU)
    exit(0)
    dst = DestinationNeuron(context, activation=None)
    src.connect_to(neuron)
    neuron.connect_to(dst)
    assert len(src.axons) == 1
    dst.dendrites[0].die()
    assert len(src.axons) == 0
    assert context.health_stat.dangle_neurons == 1  # dst


def test_system_die_all():
    # simple_log.level = simple_log.LogLevel.DEBUG
    train_x, train_y = datasets.get_odd_2()
    network = nets.create_livenet_odd_2()
    network.context.regularization_l1 = 1.  # big L1 regularization alpha will lead to quick death, even with big 'b'
    batch_iterator = gen_utils.batch_iterator(train_x, train_y, batch_size=len(train_x))
    criterion = nets.criterion_classification_n
    optimizer = optimizers.LiveNetOptimizer(network, lr=0.02)
    trainer = net_trainer.NetTrainer(network, batch_iterator, criterion, optimizer, epoch_size=100)
    trainer.step(501)
    assert network.context.health_stat.dangle_neurons == 2
    assert len(network.inputs[0].axons) == 0
    assert len(network.outputs[0].dendrites) == 0
    assert len(network.outputs[1].dendrites) == 0


if __name__ == "__main__":
    test_die()