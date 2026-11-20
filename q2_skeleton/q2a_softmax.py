import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        # Subtract row max for numerical stability
        # Keep (N, 1) shape for broadcasting
        x_max = np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x - x_max)
        x = x_exp / np.sum(x_exp, axis=1, keepdims=True)
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        # Subtract max for numerical stability
        x_exp = np.exp(x - np.max(x))
        x = x_exp / (np.sum(x_exp))
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def your_softmax_test():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    ### YOUR OPTIONAL CODE HERE
    # Check shift invariance and numerical stability with large offsets
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(softmax(x), softmax(x + 10000))
    assert np.allclose(softmax(x), softmax(x - 10000))

    # Equal inputs should give a uniform distribution
    same = softmax(np.array([4.0, 4.0, 4.0, 4.0]))
    assert np.allclose(same, np.array([0.25, 0.25, 0.25, 0.25]))

    # Single element vector
    single = softmax(np.array([10.0]))
    assert np.allclose(single, np.array([1.0]))

    # Check that rows in a random matrix sum to 1 and have valid probabilities
    np.random.seed(42)
    rand_mat = np.random.randn(10, 5) * 100
    probs = softmax(rand_mat)
    assert probs.shape == rand_mat.shape
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert np.allclose(np.sum(probs, axis=1), np.ones(10))

    print("Custom tests passed!")
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    your_softmax_test()
