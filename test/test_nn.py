import numpy as np
import pytest
from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs


def _make_simple_nn(loss='mse', seed=42):
    """
    Helper to create a simple 2-layer network for testing.
    Architecture: 4 (input) -> 3 (hidden, relu) -> 1 (output, sigmoid)
    This is intentionally small so tests run quickly and results are easy to verify.
    """
    arch = [
        {'input_dim': 4, 'output_dim': 3, 'activation': 'relu'},
        {'input_dim': 3, 'output_dim': 1, 'activation': 'sigmoid'},
    ]
    return NeuralNetwork(
        nn_arch=arch,
        lr=0.01,
        seed=seed,
        batch_size=4,
        epochs=10,
        loss_function=loss,
    )


def test_single_forward():
    """
    test that a single layer computes Z = W * A_prev + b and then applies the activation function correctly
    """
    nn = _make_simple_nn()

    # random weight (2x2) and bias (2x1) vectors
    W = np.array([[0.1, 0.2], [0.3, 0.4]])
    b = np.array([[0.0], [0.0]])
    A_prev = np.array([[1.0], [2.0]])

    # test ReLU activation
    # Z = [[0.1*1 + 0.2*2], [0.3*1 + 0.4*2]] = [[0.5], [1.1]]
    # bc both values are positive, ReLU(Z) = Z
    A_curr, Z_curr = nn._single_forward(W, b, A_prev, 'relu')
    expected_Z = np.array([[0.5], [1.1]])
    np.testing.assert_array_almost_equal(Z_curr, expected_Z)
    np.testing.assert_array_almost_equal(A_curr, expected_Z)  

    # test sigmoid activation
    # Z should be the same, but A = sigmoid(Z) != Z
    A_curr, Z_curr = nn._single_forward(W, b, A_prev, 'sigmoid')
    np.testing.assert_array_almost_equal(Z_curr, expected_Z)
    expected_A = 1 / (1 + np.exp(-expected_Z))  # sigmoid(0.5) ≈ 0.622, sigmoid(1.1) ≈ 0.750
    np.testing.assert_array_almost_equal(A_curr, expected_A)

    # test bias is correctly added to the linear transform
    b2 = np.array([[0.1], [-0.1]])
    A_curr, Z_curr = nn._single_forward(W, b2, A_prev, 'relu')
    expected_Z2 = np.array([[0.6], [1.0]])  # [0.5+0.1, 1.1-0.1]
    np.testing.assert_array_almost_equal(Z_curr, expected_Z2)


def test_forward():
    """
    test that a full forward pass produces correct output shapes, caches all intermediate values, and uses the correct output activation (ie sigmoid gives values between [0,1]).
    """
    nn = _make_simple_nn()
    X = np.random.RandomState(42).randn(5, 4)  # 5 samples, 4 features each

    output, cache = nn.forward(X)

    # output shape should be (output_dim, batch_size) = (1, 5)
    assert output.shape == (1, 5)

    # cache should contain A0, plus A and Z for each of the 2 layers
    assert 'A0' in cache  # input activation 
    assert 'A1' in cache  # hidden layer activation
    assert 'A2' in cache  # output layer activation
    assert 'Z1' in cache  # hidden layer linear transform
    assert 'Z2' in cache  # output layer linear transform

    # A0 should be the transposed input matrix
    np.testing.assert_array_almost_equal(cache['A0'], X.T)

    # output layer uses sigmoid
    assert np.all(output >= 0) and np.all(output <= 1)


def test_single_backprop():
    """
    test that a single layers backward pass produces gradients with the correct shapes for dA_prev, dW, and db. check both ReLU and sigmoid activations
    """
    nn = _make_simple_nn()
    rng = np.random.RandomState(42)

    # make random matrices with known shapes for a layer with 4 inputs, 3 outputs
    # batch of 5 samples
    W_curr = rng.randn(3, 4) # weight matrix 
    b_curr = rng.randn(3, 1) # bias vector
    Z_curr = rng.randn(3, 5) # linear transform
    A_prev = rng.randn(4, 5) # previous layer activation
    dA_curr = rng.randn(3, 5) # upstream gradient

    # ReLU activation
    dA_prev, dW_curr, db_curr = nn._single_backprop(
        W_curr, b_curr, Z_curr, A_prev, dA_curr, 'relu'
    )

    # check gradient shapes match their corresponding parameters/activations:
    assert dA_prev.shape == A_prev.shape
    assert dW_curr.shape == W_curr.shape
    assert db_curr.shape == b_curr.shape

    # sigmoid activation 
    dA_prev_sig, dW_curr_sig, db_curr_sig = nn._single_backprop(
        W_curr, b_curr, Z_curr, A_prev, dA_curr, 'sigmoid'
    )
    assert dA_prev_sig.shape == A_prev.shape
    assert dW_curr_sig.shape == W_curr.shape
    assert db_curr_sig.shape == b_curr.shape


def test_predict():
    """
    test that predict returns output with the correct shape, values are in valid range for sigmoid output
    """
    nn = _make_simple_nn()
    X = np.random.RandomState(42).randn(5, 4)  # 5 samples, 4 features
    y_hat = nn.predict(X)

    # output shape should be (output_dim, batch_size) = (1, 5)
    assert y_hat.shape == (1, 5)

    # bc of sigmoid output layer, predictions should be between 0, 1
    assert np.all(y_hat >= 0) and np.all(y_hat <= 1)

    # should be deterministic
    y_hat2 = nn.predict(X)
    np.testing.assert_array_almost_equal(y_hat, y_hat2)


def test_binary_cross_entropy():
    """
    test that: good predictions give very low loss, bad predictions give high loss, and loss is always positive
    """
    nn = _make_simple_nn(loss='bce')

    # good prediction: y_hat ≈ y 
    y = np.array([[1, 0, 1, 0]])
    y_hat = np.array([[0.999, 0.001, 0.999, 0.001]])
    loss = nn._binary_cross_entropy(y, y_hat)
    assert loss < 0.01  # small loss for good predictions

    # bad prediction: y_hat ≈ 1 - y 
    y_hat_bad = np.array([[0.001, 0.999, 0.001, 0.999]])
    loss_bad = nn._binary_cross_entropy(y, y_hat_bad)
    assert loss_bad > 5.0  # High loss

    # BCE loss is always non-negative
    assert loss >= 0
    assert loss_bad >= 0


def test_binary_cross_entropy_backprop():
    """
    test the gradient direction. When y=1, gradient should be negative. When y=0, gradient should be positive 
    """
    nn = _make_simple_nn(loss='bce')
    y = np.array([[1, 0, 1]])
    y_hat = np.array([[0.8, 0.3, 0.9]])
    dA = nn._binary_cross_entropy_backprop(y, y_hat)

    assert dA.shape == y_hat.shape

    # y=1, dL/dA = -1/y_hat
    # ie loss would increase if y_hat decreased, so gradient descent will push y_hat toward 1
    assert dA[0, 0] < 0
    assert dA[0, 2] < 0

    # y=0, dL/dA = 1/(1-y_hat). ie the loss would increase if y_hat increased, so gradient descent will push y_hat downward toward 0
    assert dA[0, 1] > 0


def test_mean_squared_error():
    """
    test that perfect predictions give exactly zero loss, known inputs produce the expected loss, and loss is always positive
    """
    nn = _make_simple_nn(loss='mse')

    # Perfect prediction y_hat == y
    y = np.array([[1.0, 2.0, 3.0]])
    y_hat = np.array([[1.0, 2.0, 3.0]])
    loss = nn._mean_squared_error(y, y_hat)
    assert loss == pytest.approx(0.0, abs=1e-10)

    # known calc: y=[1, 0], y_hat=[0, 1]
    # MSE = [(1-0)^2 + (0-1)^2] / 2 = [1 + 1] / 2 = 1.0
    y2 = np.array([[1.0, 0.0]])
    y_hat2 = np.array([[0.0, 1.0]])
    loss2 = nn._mean_squared_error(y2, y_hat2)
    assert loss2 == pytest.approx(1.0, abs=1e-10)

    # MSE loss is always non-negative
    assert loss >= 0
    assert loss2 >= 0


def test_mean_squared_error_backprop():
    """
    test gradient direction. When y_hat < y, gradient should be negative. When y_hat > y, gradient should be positive. When y_hat == y, gradient should be exactly zero
    """
    nn = _make_simple_nn(loss='mse')
    y = np.array([[1.0, 0.0, 0.5]])
    y_hat = np.array([[0.8, 0.3, 0.5]])
    dA = nn._mean_squared_error_backprop(y, y_hat)

    assert dA.shape == y.shape

    # y_hat=0.8 < y=1.0
    # dL/dA = 2*(0.8-1.0)/3 < 0 
    assert dA[0, 0] < 0

    # y_hat=0.3 > y=0.0 
    # dL/dA = 2*(0.3-0.0)/3 > 0 
    assert dA[0, 1] > 0

    # y_hat=0.5 == y=0.5 
    # dL/dA = 2*(0.5-0.5)/3 = 0 
    assert dA[0, 2] == pytest.approx(0.0, abs=1e-10)


def test_sample_seqs():
    """
    test the oversampling correctly balances classes   
 """
    seqs = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF']
    labels = [True, True, False, False, False, False]

    np.random.seed(42)
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    # we should end up with equal positives and negatives after balancing
    n_pos = sum(sampled_labels)
    n_neg = len(sampled_labels) - n_pos
    assert n_pos == n_neg  

    # total = 2 * max(2, 4) = 8 (4 positives + 4 negatives)
    assert len(sampled_seqs) == 8
    assert len(sampled_labels) == 8

    # all positive labeled sequences should come from the original positive set
    pos_sampled = [s for s, l in zip(sampled_seqs, sampled_labels) if l]
    for s in pos_sampled:
        assert s in ['AAA', 'BBB']

    # all negative labeled sequences should come from the original negative set
    neg_sampled = [s for s, l in zip(sampled_seqs, sampled_labels) if not l]
    for s in neg_sampled:
        assert s in ['CCC', 'DDD', 'EEE', 'FFF']


def test_one_hot_encode_seqs():
    """
    test the one-hot encoding against known values
    A->[1,0,0,0], T->[0,1,0,0], C->[0,0,1,0], G->[0,0,0,1]
    """
    # test simple 3 nucleotide sequence: AGA
    # Expected: [1,0,0,0] + [0,0,0,1] + [1,0,0,0] = [1,0,0,0,0,0,0,1,1,0,0,0]
    seqs = ['AGA']
    encoded = one_hot_encode_seqs(seqs)
    expected = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]])
    np.testing.assert_array_equal(encoded, expected)

    # test all four nucleotides: ATCG
    # Expected: [1,0,0,0] + [0,1,0,0] + [0,0,1,0] + [0,0,0,1]
    seqs2 = ['ATCG']
    encoded2 = one_hot_encode_seqs(seqs2)
    expected2 = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
    np.testing.assert_array_equal(encoded2, expected2)

    # test multiple sequences: output shape should be (num_seqs, seq_len * 4)
    seqs3 = ['AT', 'CG']
    encoded3 = one_hot_encode_seqs(seqs3)
    assert encoded3.shape == (2, 8)  # 2 sequences, each 2 nucleotides * 4 = 8
    np.testing.assert_array_equal(encoded3[0], [1, 0, 0, 0, 0, 1, 0, 0])  # AT
    np.testing.assert_array_equal(encoded3[1], [0, 0, 1, 0, 0, 0, 0, 1])  # CG

    assert isinstance(encoded, np.ndarray)