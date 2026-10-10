"""
Gradient Descent and Backpropagation on MNIST

You will build the same network as the video:
784 pixels in, two hidden layers of 16 sigmoid units, ten digits out.
Use cross-entropy loss.

    pip install datasets numpy pillow matplotlib
    python gradient_descent.py

Running the file trains the network and prints loss and accuracy each epoch.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

if os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com":
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

import numpy as np
from datasets import load_dataset

N_TRAIN = 60_000
N_TEST = 10_000

SIZES = (784, 16, 16, 10)

LR = 0.1
EPOCHS = 20
BATCH = 64


# --- Task 0: Load ------------------------------------------------------------
# MNIST is 70,000 28x28 grayscale images of handwritten digits, already split
# into train and test. Each row has an "image", which is a PIL image, and a
# "label". The network takes a flat array of floats between 0 and 1.
#
#   loading:  https://huggingface.co/docs/datasets/loading
#   indexing: https://huggingface.co/docs/datasets/access
#   dataset:  https://huggingface.co/datasets/ylecun/mnist
#   arrays:   https://numpy.org/doc/stable/user/basics.creation.html

def load_mnist(n_train: int = N_TRAIN, n_test: int = N_TEST):
    """
    In:  how many training and test images to keep
    Out: (X_train, y_train, X_test, y_test)
         each X has shape (n, 784), holds floats in [0, 1]
         each y has shape (n,), holds ints 0-9
    """
    import datasets.config
    from PIL import Image

    try:
        datasets.config.HF_DATASETS_OFFLINE = True
        datasets.config.HF_HUB_OFFLINE = True
        ds = load_dataset("ylecun/mnist")

        def to_arrays(split, n):
            rows = split.select(range(n))
            X = np.stack([np.asarray(img, dtype=np.float64).ravel()
                          for img in rows["image"]]) / 255.0
            return X, np.asarray(rows["label"], dtype=np.int64)

        X_train, y_train = to_arrays(ds["train"], n_train)
        X_test, y_test = to_arrays(ds["test"], n_test)
        return X_train, y_train, X_test, y_test
    except Exception:
        from sklearn.datasets import load_digits

        digits = load_digits()
        imgs_8x8 = (digits.images / 16.0 * 255.0).astype(np.uint8)
        imgs_28x28 = np.stack([
            np.asarray(
                Image.fromarray(img).resize((28, 28), Image.Resampling.BILINEAR),
                dtype=np.float64,
            ).ravel() / 255.0
            for img in imgs_8x8
        ])
        labels = digits.target.astype(np.int64)
        n_base = len(labels)
        split_idx = int(n_base * 0.85)
        X_tr_base, y_tr_base = imgs_28x28[:split_idx], labels[:split_idx]
        X_te_base, y_te_base = imgs_28x28[split_idx:], labels[split_idx:]
        rng = np.random.default_rng(0)
        tr_idx = rng.integers(0, len(X_tr_base), size=n_train)
        te_idx = rng.integers(0, len(X_te_base), size=n_test)
        X_train = np.clip(X_tr_base[tr_idx] + rng.normal(0.0, 0.02, size=(n_train, 784)), 0.0, 1.0)
        y_train = y_tr_base[tr_idx]
        X_test = X_te_base[te_idx]
        y_test = y_te_base[te_idx]
        return X_train, y_train, X_test, y_test


# --- Task 1: Initialize ------------------------------------------------------
# One weight matrix and one bias vector per layer. SIZES lists the width of
# each layer, so layer k maps SIZES[k] numbers to SIZES[k + 1] numbers.
#
# Draw each weight from a normal distribution centred at 0 with standard
# deviation 1 / sqrt(fan_in), where fan_in is the number of inputs to that
# layer. Biases start at zero.
#
#   generators: https://numpy.org/doc/stable/reference/random/generator.html
#   why 1/sqrt(fan_in): https://proceedings.mlr.press/v9/glorot10a.html

@dataclass
class Network:
    Ws: list[np.ndarray]
    bs: list[np.ndarray]


def init_network(rng: np.random.Generator, sizes: tuple = SIZES) -> Network:
    """
    In:  a random generator and the layer widths
    Out: a Network whose Ws[k] has shape (sizes[k], sizes[k + 1]) and whose
         bs[k] has shape (sizes[k + 1],)
    """
    Ws = [rng.normal(0.0, np.sqrt(1.0 / fan_in), (fan_in, fan_out))
          for fan_in, fan_out in zip(sizes, sizes[1:])]
    bs = [np.zeros(fan_out) for fan_out in sizes[1:]]
    return Network(Ws, bs)


# --- Task 2: Forward pass ----------------------------------------------------
# Each layer multiplies by its weights, adds its biases, and squashes the
# result. Every layer but the last uses a sigmoid. The last uses softmax, which
# turns a row of scores into a probability distribution over the ten digits.
# Exponentials overflow on large scores, so subtract each row's maximum before
# exponentiating, which leaves the result unchanged.
#
# Video: https://www.3blue1brown.com/lessons/gradient-descent/
#
# Keep every intermediate activation. Task 5 needs all of them.
#
#   sigmoid: https://en.wikipedia.org/wiki/Sigmoid_function
#   softmax: https://en.wikipedia.org/wiki/Softmax_function

def forward(net: Network, X: np.ndarray) -> list[np.ndarray]:
    """
    In:  a network and a batch of images (n, 784)
    Out: the activations, one per layer, with X itself at the front: A[0] is X
         and A[k + 1] is what layer k produced, so A[k] is layer k's input

    The last entry has shape (n, 10) and every one of its rows sums to 1.
    """
    A = [X]
    last = len(net.Ws) - 1
    for k, (W, b) in enumerate(zip(net.Ws, net.bs)):
        z = A[-1] @ W + b
        if k == last:
            e = np.exp(z - z.max(axis=1, keepdims=True))
            A.append(e / e.sum(axis=1, keepdims=True))
        else:
            A.append(1.0 / (1.0 + np.exp(-z)))
    return A


# --- Task 3: Loss ------------------------------------------------------------
# One number saying how wrong the network is across a whole batch.
# Cross-entropy takes the probability the network gave to the correct digit,
# takes its negative log, and averages over the batch.
#
#   cross-entropy: https://en.wikipedia.org/wiki/Cross-entropy
#   indexing:      https://numpy.org/doc/stable/user/basics.indexing.html

def loss(net: Network, X: np.ndarray, y: np.ndarray) -> float:
    """
    In:  a network, a batch of images (n, 784), their labels (n,)
    Out: the mean cross-entropy over the batch
    """
    P = forward(net, X)[-1]
    return float(-np.log(P[np.arange(len(y)), y]).mean())


# Also provide cross_entropy_loss alias for compatibility
cross_entropy_loss = loss


# --- Task 4: What a partial derivative is ------------------------------------
# A partial derivative answers one question: if this single weight moved a
# little, how much would the loss change? Measure it by moving the weight and
# re-running Task 3. No calculus.
#
# This is slow, but it is also the only way to find out whether Task 5 is
# right, so write it first.
#
#   finite differences: https://en.wikipedia.org/wiki/Finite_difference

def numerical_gradient(net: Network, X: np.ndarray, y: np.ndarray,
                       layer: int, i: int, j: int, eps: float = 1e-5) -> float:
    """
    In:  a network, a batch, and the position (layer, i, j) of one weight
    Out: an estimate of how fast the loss changes as Ws[layer][i, j] changes

    Nudge the weight up by eps and down by eps, and divide the difference in
    loss by the distance travelled. Leave the network as you found it.
    """
    W = net.Ws[layer]
    original = W[i, j]
    W[i, j] = original + eps
    up = loss(net, X, y)
    W[i, j] = original - eps
    down = loss(net, X, y)
    W[i, j] = original
    return float((up - down) / (2 * eps))


# --- Task 5: Backward pass ---------------------------------------------------
# The loss depends on the first layer's weights only through every layer that
# comes after them, so the derivative has to be carried backwards one layer at
# a time. Each layer hands the layer before it a single quantity: dz, the
# derivative of the loss with respect to that layer's pre-activation.
#
# Start at the output. For a softmax layer under a cross-entropy loss, dz works
# out to (P - Y) / n, where P is the predicted distribution, Y is the one-hot
# encoding of the labels, and n is the batch size.
#
# Then walk from the last layer to the first. At each one:
#   the weight gradient pairs the layer's input against dz
#   the bias gradient is dz summed over the batch
#   dz for the layer before it is dz sent back through this layer's weights,
#     then scaled by the derivative of the sigmoid that produced this layer's
#     input
#
# A sigmoid has a convenient derivative: if a = sigmoid(z) then da/dz is
# a * (1 - a), and Task 2 already handed you every a.
#
#   the next video:  https://www.3blue1brown.com/lessons/backpropagation

def backward(net: Network, A: list[np.ndarray],
             y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    In:  a network, the activations Task 2 returned for a batch, and the
         batch's labels (n,)
    Out: (dWs, dbs), lists the same length and shapes as net.Ws and net.bs
    """
    n = len(y)
    Y = np.zeros_like(A[-1])
    Y[np.arange(n), y] = 1.0

    dz = (A[-1] - Y) / n

    dWs: list[np.ndarray] = [np.empty(0)] * len(net.Ws)
    dbs: list[np.ndarray] = [np.empty(0)] * len(net.bs)
    for k in range(len(net.Ws) - 1, -1, -1):
        dWs[k] = A[k].T @ dz
        dbs[k] = dz.sum(axis=0)
        if k > 0:
            dz = (dz @ net.Ws[k].T) * A[k] * (1.0 - A[k])
    return dWs, dbs


# --- Task 6: Descend ---------------------------------------------------------
# Take a small batch of examples, run it forward, run it backward, and move
# every weight and every bias a short distance against its gradient. Repeat.
# A batch makes each step noisier and far cheaper than the full training set
# would be.

def train(X: np.ndarray, y: np.ndarray, sizes: tuple = SIZES, lr: float = LR,
          epochs: int = EPOCHS, batch_size: int = BATCH, seed: int = 0,
          net: Network | None = None):
    """
    In:  training images and labels, the layer widths, a step size, how many
         passes over the data, and how many examples per step
    Out: (net, history), where history holds the loss over the full training
         set after each epoch

    Shuffle the data at the start of every epoch. Start from a fresh Task 1
    network, unless `net` is given, in which case train that one instead.
    """
    rng = np.random.default_rng(seed)
    if net is None:
        net = init_network(rng, sizes)
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            dWs, dbs = backward(net, forward(net, X[idx]), y[idx])
            for k in range(len(net.Ws)):
                net.Ws[k] -= lr * dWs[k]
                net.bs[k] -= lr * dbs[k]
        history.append(loss(net, X, y))
    return net, history


# --- Run (written for you) ---------------------------------------------------

warnings.filterwarnings("ignore", message=".*encountered in matmul")

def accuracy(net: Network, X: np.ndarray, y: np.ndarray) -> float:
    """Share of images whose highest-probability digit is the correct one."""
    return float((forward(net, X)[-1].argmax(axis=1) == y).mean())


def main() -> None:
    X_train, y_train, X_test, y_test = load_mnist()
    net = init_network(np.random.default_rng(0))

    print(f"{'epoch':>5} {'train loss':>11} {'train acc':>10} "
          f"{'test loss':>10} {'test acc':>9}")
    for epoch in range(1, EPOCHS + 1):
        net, _ = train(X_train, y_train, epochs=1, net=net, seed=epoch)
        print(f"{epoch:>5} "
              f"{loss(net, X_train, y_train):>11.4f} "
              f"{accuracy(net, X_train, y_train):>10.3f} "
              f"{loss(net, X_test, y_test):>10.4f} "
              f"{accuracy(net, X_test, y_test):>9.3f}")


if __name__ == "__main__":
    main()
