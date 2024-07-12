import torch

from .core.livenet import Context, RegularNeuron, DestinationNeuron, SourceNeuron, DataNeuron
from .core import optimizers, livenet
from . import nets, gen_utils, net_trainer
from . import datasets
from ai_libs.simple_log import LOG


def test_die():
    module = torch.nn.Module()
    context = Context(module, seed=42)
    context.module = torch.nn.Module()
    src = DataNeuron(context)
    src.set_output(torch.Tensor(42))
    neuron = RegularNeuron(context, activation=torch.nn.ReLU)
    return
    dst = DestinationNeuron(context, activation=None)
    src.connect_to(neuron)
    neuron.connect_to(dst)
    assert len(src.axons) == 1
    dst.dendrites[0].die()
    assert len(src.axons) == 0
    assert context.topology_stat.dangle_neurons == 1  # dst


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
    assert len(network.inputs[0].axons) == 0
    assert len(network.outputs[0].dendrites) == 0
    assert len(network.outputs[1].dendrites) == 0
    stat = network.context.topology_stat.get_stat()
    assert stat["dangle"]["RegularNeuron"] == 0
    assert stat["dangle"]["DestinationNeuron"] == 2
    assert stat["useless"]["RegularNeuron"] == 0
    assert stat["useless"]["DataNeuron"] == 1


# def test_system():
#     # core.simple_log.level = core.simple_log.LogLevel.DEBUG
#     train_x, train_y = datasets.get_odd_2()
#     context = livenet.Context()
#     context.liveness_die_after_n_sign_changes = 2
#     context.regularization_l1 = 0.01
#     network = nets.create_livenet_odd_2(context)
#     batch_iterator = gen_utils.batch_iterator(train_x, train_y, batch_size=len(train_x))
#     criterion = nets.criterion_classification_n
#     optimizer = optimizers.LiveNetOptimizer(network, lr=0.05)
#     trainer = net_trainer.NetTrainer(network, batch_iterator, criterion, optimizer, epoch_size=100)
#     trainer.step(401)
#     assert len(trainer.history[0]["params"]) > len(trainer.history[-1]["params"])  # some stuff must be dead
#     assert trainer.history[0]["loss"] > 0.03
#     assert trainer.history[-1]["loss"] < 0.01
#     assert trainer.history[-1]["loss_reg"] > 0.0


def _build_symmetric_dangle_net():
    net = livenet.LiveNet()
    net.outputs += [DestinationNeuron(net.context, activation=None), DestinationNeuron(net.context, activation=None)]
    net.inputs += [DataNeuron(net.context)]
    neuron = RegularNeuron(net.context, activation=torch.nn.ReLU())
    neuron.connect_to(net.outputs[0])
    neuron.connect_to(net.outputs[1])
    # net.inputs[0].connect_to(neuron)
    net.root.visit("init_weight")
    # with torch.no_grad():
    #     net.outputs[0].b[...] = 14.0
    #     net.outputs[1].b[...] = -14.1
    #     neuron.b[...] = 20.0
    #     neuron.axons[0].k[...] = 10.0
    #     neuron.axons[1].k[...] = -10.1
    return net


def test_dangle_symmetric_die():
    train_x, train_y = datasets.get_odd_2()
    network = _build_symmetric_dangle_net()
    # train_x is not actually used as networks input doesn't connected to anything
    batch_iterator = gen_utils.batch_iterator(train_x, train_y, batch_size=3)
    criterion = nets.criterion_classification_n
    optimizer = nets.create_optimizer(network)
    trainer = net_trainer.NetTrainer(network, batch_iterator, criterion, optimizer, epoch_size=100)
    network.context.regularization_l1 = 0.001
    optimizer.learning_rate = 0.01
    trainer.step(5000)
    assert len(list(network.parameters())) == 2
