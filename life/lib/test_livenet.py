import torch

from life.lib.livenet import Context, RegularNeuron, DestinationNeuron


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
