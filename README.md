## Personal research project on AI basics, using CIFAR10 as a toy dataset
Summary: Attempt to classify CIFAR10 with as small amount of operations as possible.
Unused single neurons' death (V1) and sparse conv layers with dynamic computational complexity (V2) are examined and employed.

Worked on in my own time between 2022 and 2025: V1 mostly in 2022-2024, V2 in 2025.

### V1. Dynamically killing unnecessary neurons
#### Goal: Examine death of unnecessary neurons in flexible "any topology" per-single neuron graph.
Implemented per-single-neuron dynamic graph with capability of death of useless neurons, dynamic birth was also in mind, but never implemented.
Neurons were brought to "useless" state by imposing reasonable L1 regularization. Those which started to jump around zero, eventually die in course of training.
With this approach, I was able to obtain 2-layer perceptron, with 40 neurons in total, which classifies CIFAR10 with 30% accuracy
(number from a run at the time, not re-verified since).

After a while I decided that implementing larger convolutional classifier using per-single-neuron graph is not practical and moved to V2.

### V2. Smart custom convolutional layers
#### Goal: Classify CIFAR10 dataset within as small amount of operations as possible.
Operations are counted as single scalar multiply-add, as they would run on general purpose CPU, without GEMM structure.
To achieve this, two ideas are employed:
- Death of unused connections from V1 (implemented as masks of custom conv layer)
- Dynamic computational complexity of network (for simpler parts of the image, e.g. big patch of blue sky, it would use less amount of operations)

For this I implemented a custom conv layer capable of doing self-observation and dynamic masking during its work.
Dynamic computational complexity would be efficiently achieved, as RELU outputs are often zeros, and no further processing is required at next layer for such outputs.
Given my custom conv layers are smart, they aim to L1-penalize actual operations, not weights.

V2 was left at the point where the layer and a naive reference forward pass work; the accuracy-per-operation curves it was built to measure are not done yet.
