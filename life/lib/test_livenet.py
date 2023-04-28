import torch

from life.lib.livenet import Context, RegularNeuron, DestinationNeuron
import life.lib as lib


def test_die():
    module = torch.nn.Module()
    context = Context(module, 42)
    src = RegularNeuron(context, activation=torch.nn.ReLU())
    neuron = RegularNeuron(context, activation=torch.nn.ReLU)
    dst = DestinationNeuron(context, activation=None)
    src.connect_to(neuron)
    neuron.connect_to(dst)
    assert len(src.axons) == 1
    dst.dendrites[0].die()
    assert len(src.axons) == 0
    assert context.death_stat.dangle_neurons == 1


def test_die_full_net():
    # lib.simple_log.level = lib.simple_log.LogLevel.DEBUG
    train_x, train_y = lib.datasets.get_odd()
    network = lib.nets.create_livenet_odd_2()
    batch_iterator = lib.gen_utils.batch_iterator(train_x, train_y, batch_size=len(train_x))
    criterion = lib.nets.criterion_n
    optimizer = lib.optimizer.LiveNetOptimizer(network, lr=0.01)
    trainer = lib.trainer.Trainer(network, batch_iterator, criterion, optimizer, epoch_size=100)
    trainer.step(2)
    assert network.context.death_stat.dangle_neurons == 2
    assert len(network.inputs[0].axons) == 0
    assert len(network.outputs[0].dendrites) == 0
    assert len(network.outputs[1].dendrites) == 0
