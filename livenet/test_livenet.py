import torch

import numpy as np
from .backend.core import Context, RegularNeuron, DestinationNeuron, SourceNeuron, InputNeuron
from .backend import optimizers, core
from . import nets, gen_utils, net_trainer
from . import datasets
from ai_libs.simple_log import LOG
import livenet


def test_die():
    module = torch.nn.Module()
    context = Context(module, seed=42)
    context.module = torch.nn.Module()
    src = InputNeuron(context)
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
    network.context.regularization_l1 = 1.  # big L1 regularization alpha will lead to quick death of all neurons
    batch_iterator = gen_utils.batch_iterator(train_x, train_y, batch_size=len(train_x))
    criterion = nets.criterion_classification_n
    optimizer = optimizers.optimizers.LiveNetOptimizer(network, lr=0.02)
    trainer = net_trainer.NetTrainer(network, batch_iterator, criterion, optimizer, epoch_size=100)
    trainer.step(501)
    assert len(network.inputs[0].axons) == 0
    assert len(network.outputs[0].dendrites) == 0
    assert len(network.outputs[1].dendrites) == 0
    stat = network.context.topology_stat.get_stat()
    assert stat["dangle"]["RegularNeuron"] == 0
    assert stat["dangle"]["DestinationNeuron"] == 2
    assert stat["useless"]["RegularNeuron"] == 0
    assert stat["useless"]["InputNeuron"] == 1


def _build_symmetric_dangle_net():
    net = livenet.backend.core.LiveNet()
    net.outputs += [DestinationNeuron(net.context, activation=None), DestinationNeuron(net.context, activation=None)]
    net.inputs += [InputNeuron(net.context)]
    neuron = RegularNeuron(net.context, activation=torch.nn.ReLU())
    neuron.connect_to(net.outputs[0])
    neuron.connect_to(net.outputs[1])
    # net.inputs[0].connect_to(neuron)
    net.root.visit_member("init_weight")
    with torch.no_grad():
        net.outputs[0].b[...] = -30.0
        net.outputs[1].b[...] = -0.0
        neuron.b[...] = 6.0
        neuron.axons[0].k[...] = 2.0
        neuron.axons[1].k[...] = 0.0
    return net


def test_dangle_symmetric_die():
    train_x, train_y = datasets.get_odd_2()
    network = _build_symmetric_dangle_net()
    # train_x is not actually used as networks input doesn't connected to anything
    batch_iterator = gen_utils.batch_iterator(train_x, train_y, batch_size=3)
    criterion = nets.criterion_classification_n
    optimizer = nets.create_optimizer(network)
    trainer = net_trainer.NetTrainer(network, batch_iterator, criterion, optimizer, epoch_size=100)
    network.context.regularization_l1 = 0.05
    optimizer.learning_rate = 0.002
    trainer.step(15000)
    assert len(list(network.parameters())) == 2


def test_odd():
    # simple_log.level = simple_log.LogLevel.DEBUG
    train_x, train_y = datasets.get_odd_2()
    network = nets.create_livenet_odd_2()
    assert len(list(network.parameters())) == 10
    res = network(train_x)
    network.context.regularization_l1 = 0.05
    batch_iterator = gen_utils.batch_iterator(train_x, train_y, batch_size=len(train_x))
    criterion = nets.criterion_classification_n
    optimizer = livenet.backend.optimizers.optimizers.LiveNetOptimizer(network, lr=0.01)
    trainer = net_trainer.NetTrainer(network, batch_iterator, criterion, optimizer, epoch_size=50)
    trainer.step(1001)
    scores = torch.nn.functional.softmax(network(train_x), dim=1).detach().numpy()
    # LOG(scores)
    prediction = np.argmax(scores, axis=1)
    assert np.all(prediction == train_y.numpy().squeeze(1))
    assert len(list(network.parameters())) == 8
    # backend.livenet.export_onnx(network, "/home/spometun/table/home/net.onnx")


def test_mnist_perceptron_die():
    # simple_log.level = simple_log.LogLevel.DEBUG
    downscale = (14, 14)
    train_x, train_y = datasets.to_plain(*datasets.get_mnist_train(), downscale=downscale, to_odd=True,
                                         to_gray=True)
    network = nets.create_perceptron(train_x.shape[1], 2, 2)
    batch_iterator = gen_utils.batch_iterator(train_x, train_y, batch_size=1000)
    criterion = nets.criterion_classification_n
    optimizer = nets.create_optimizer(network)
    trainer = net_trainer.NetTrainer(network, batch_iterator, criterion, optimizer, epoch_size=50)
    assert len(list(network.parameters())) == 16
    network.context.regularization_l1 = 0.001
    optimizer.learning_rate = 0.01
    trainer.step(500)

    assert len(list(network.parameters())) == 6
    pred = network(train_x)
    pred_bin = np.argmax(pred.detach().numpy(), axis=1, keepdims=True)
    diff = train_y.numpy() - pred_bin
    accuracy = len(diff[diff == 0]) / len(diff)
    assert accuracy > 0.7
