#!/usr/bin/env python

import json
import sys
import time

import matplotlib
import matplotlib.pyplot as plt

from helpers.treebank import StanfordSentiment
from q2c_word2vec import *
from q2d_sgd import *

# Check Python Version
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

matplotlib.use('agg')

# Reset the random seed to make sure that everyone gets the same results
random.seed(1987)
dataset = StanfordSentiment()
tokens = dataset.tokens()
n_words = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dim_vectors = 10

# Context size
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(1407)
np.random.seed(1411)

start_time = time.time()
word_vectors = np.concatenate(
    ((np.random.rand(n_words, dim_vectors) - 0.5) /
     dim_vectors, np.zeros((n_words, dim_vectors))),
    axis=0)
initial_vectors = word_vectors.copy()
word_vectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, neg_sampling_loss_and_gradient),
    word_vectors, 0.3, 40000, None, True, print_every=10)
# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.

print("sanity check: cost at convergence should be around or below 10")
print("training took {} seconds".format(time.time() - start_time))

# concatenate the input and output word vectors
word_vectors = np.concatenate(
    (word_vectors[:n_words, :], word_vectors[n_words:, :]),
    axis=1)
initial_vectors = np.concatenate(
    (initial_vectors[:n_words, :], initial_vectors[n_words:, :]),
    axis=1)

visualize_words = [
    "great", "cool", "brilliant", "wonderful", "excellent", "amazing", "funny", "sweet", "enjoyable",
    "boring", "bad", "dumb", "annoying", "terrible", "disappointing", "awful", "horrible", "dull",
    "female", "male", "queen", "king", "man", "woman",
    "rain", "snow", "hail",
    "coffee", "tea",
    "couch", "table", "chair", "rug"]

visualize_idx = [tokens[word] for word in visualize_words]

def plot_words(ax, vectors, title):
    visualize_vecs = vectors[visualize_idx, :]
    temp = (visualize_vecs - np.mean(visualize_vecs, axis=0))
    covariance = 1.0 / len(visualize_idx) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp.dot(U[:, 0:2])
    for i in range(len(visualize_words)):
        ax.text(coord[i, 0], coord[i, 1], visualize_words[i],
                bbox=dict(facecolor='blue', alpha=0.1))
    ax.set_xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    ax.set_ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_words(axes[0], initial_vectors, "Before training")
plot_words(axes[1], word_vectors, "After training")
plt.tight_layout()
plt.savefig('word_vectors.png')

# Export the average of the center and outside vectors for reuse in Section 4
# (training a DAN with your own embeddings instead of GloVe).
center_vectors = word_vectors[:, :dim_vectors]
outside_vectors = word_vectors[:, dim_vectors:]
average_vectors = (center_vectors + outside_vectors) / 2
np.save("word2vec_vectors.npy", average_vectors)
vocab_export = dict(tokens)
vocab_export["unk"] = vocab_export["UNK"]  # match DAN.ipynb's lowercase key
with open("word2vec_vocab.json", "w") as f:
    json.dump(vocab_export, f)
