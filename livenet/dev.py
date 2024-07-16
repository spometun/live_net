# PLAN:
# 2. Create downscaled CIFAR network, see how it trains, death, stats, etc.
# 3. Refactor to have core + high-level structure
# may be do 3. first, at least to 3. before improving/debugging deaths on 1. and 2.
# 4. Solve Context design and/or death logic (what required to die parameters) design (what is provided into constructors) if needed
# 5. Provide OBSERVABILITY and after that/alongside with that conduct meaningful experiments
# experiments could be: create binary number dynamic addition (?!),
# or cifar recognition with born/deaths,
# add analog of batch norm (additive neuron for all dendrites, with trained additive constant)

import numpy as np
import torch.nn as nn
from . import datasets
from . import nets, gen_utils, net_trainer
import livenet

# importlib.reload(core)


def test_odd():
    # simple_log.level = simple_log.LogLevel.DEBUG
    train_x, train_y = datasets.get_odd_2()
    network = nets.create_livenet_odd_2()
    assert len(list(network.parameters())) == 10
    res = network(train_x)
    network.context.regularization_l1 = 0.05
    batch_iterator = gen_utils.batch_iterator(train_x, train_y, batch_size=len(train_x))
    criterion = nets.criterion_classification_n
    optimizer = livenet.core.optimizers.optimizers.LiveNetOptimizer(network, lr=0.01)
    trainer = net_trainer.NetTrainer(network, batch_iterator, criterion, optimizer, epoch_size=50)
    trainer.step(1001)
    scores = nn.functional.softmax(network(train_x), dim=1).detach().numpy()
    # LOG(scores)
    prediction = np.argmax(scores, axis=1)
    assert np.all(prediction == train_y.numpy().squeeze(1))
    assert len(list(network.parameters())) == 8
    # core.livenet.export_onnx(network, "/home/spometun/table/home/net.onnx")


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
    network.context.regularization_l1 = 0.01
    optimizer.learning_rate = 0.01
    trainer.step(500)

    assert len(list(network.parameters())) == 10  # 8
    pred = network(train_x)
    pred_bin = np.argmax(pred.detach().numpy(), axis=1, keepdims=True)
    diff = train_y - pred_bin
    accuracy = len(diff[diff == 0]) / len(diff)
    assert accuracy > 0.7
