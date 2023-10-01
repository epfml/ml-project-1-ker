import numpy as np

"----------------------------------------------------------------------------------------------------------------------"
"""                                         Helper functions                                                         """
"----------------------------------------------------------------------------------------------------------------------"


def load_data():
    """Load data and convert it to the metric system."""
    path_dataset = "../data/train.csv"
    dataset = np.genfromtxt(path_dataset, delimiter=",", skip_header=1)
    data = dataset[:, 2:]
    ids = dataset[:, 0]
    pred = np.genfromtxt(
        path_dataset,
        delimiter=",",
        skip_header=1,
        usecols=[0],
        converters={0: lambda x: 0 if b"s" in x else 1}
    )
    return data, pred, ids


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(data, pred):
    """Form (y,tX) to get regression data in matrix form."""
    y = pred
    x = data
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


"----------------------------------------------------------------------------------------------------------------------"
"""                                     Linear regression using gradient descent                                     """
"----------------------------------------------------------------------------------------------------------------------"


def compute_loss_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,M)
        w: numpy array of shape=(M,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - np.dot(tx, w)
    return np.dot(e.T, e)


def compute_gradient_mse(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,M)
        w: numpy array of shape=(M, ). The vector of model parameters.

    Returns:
        An numpy array of shape (M, ) (same shape as w), containing the gradient of the loss at w.
    """

    e = y - np.dot(tx, w)
    gradient = -(1 / len(y)) * np.dot(tx.T, e)

    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,M)
        initial_w: numpy array of shape=(M, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (M, ),
            for each iteration of GD
    """
    # Define parameters to store w and loss
    loss = 0
    w = initial_w
    # Gradient descent
    for n_iter in range(max_iters):
        loss = compute_loss_mse(y, tx, w)
        grad = compute_gradient_mse(y, tx, w)
        w = w - gamma * grad

    return w, loss


"----------------------------------------------------------------------------------------------------------------------"
"""                                Linear regression using stochastic gradient descent                               """
"----------------------------------------------------------------------------------------------------------------------"


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,M)
        initial_w: numpy array of shape=(M, ). The initial guess (or the initialization) for the model parameters
        batch_size: scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: scalar denoting the total number of iterations of SGD
        gamma: scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (M, ),
            for each iteration of SGD
    """

    # Define parameters to store w and loss
    final_loss = 0
    w = initial_w

    for n_iter in range(max_iters):
        loss = 0
        grad = np.zeros(w.shape)

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss = loss + compute_loss_mse(minibatch_y, minibatch_tx, w)
            grad = grad + compute_gradient_mse(minibatch_y, minibatch_tx, w)

        grad = (1 / batch_size) * grad
        final_loss = loss
        w = w - gamma * grad

    return w, final_loss


"----------------------------------------------------------------------------------------------------------------------"
"""                              Least squares regression using normal equations                                     """
"----------------------------------------------------------------------------------------------------------------------"