import numpy as np

"----------------------------------------------------------------------------------------------------------------------"
"""                                         Helper functions                                                         """
"----------------------------------------------------------------------------------------------------------------------"


def load_data(datafile):
    """Load data and convert it to the metric system."""
    path_dataset = f"../data/{datafile}"
    dataset = np.genfromtxt(path_dataset, delimiter=",", skip_header=1)
    return dataset


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def standardize_clean(x):
    """
    Replace NaN values in a feature with the mean of the non-NaN values.

    Args:
        x (numpy.ndarray): 1D array representing a feature.

    Returns:
        numpy.ndarray: 1D array with NaN values replaced by the mean.
    """
    nan_indices = np.isnan(x)
    non_nan_indices = ~nan_indices  # Invert the nan_indices to get non-NaN indices
    mean_x = np.mean(x[non_nan_indices])
    x[nan_indices] = mean_x
    
    x = x - mean_x
    std_x = np.std(x[non_nan_indices])
    if std_x != 0 : 
        x = x / std_x
    return x, std_x


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
    return (1/(2*len(y)))*np.dot(e.T, e)


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


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
    Args:
        y: shape=(N, )
        tx: shape=(N,M)
        initial_w: shape=(M, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (M, ), for each iteration         of SGD
    """

    w = initial_w
    loss = np.inf
    batch_size = 1

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            grad = compute_gradient_mse(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = compute_loss_mse(y, tx, w)
    return w, loss


"----------------------------------------------------------------------------------------------------------------------"
"""                              Least squares regression using normal equations                                     """
"----------------------------------------------------------------------------------------------------------------------"


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N,M).
    Returns:
        w: optimal weights, numpy array of shape(M,).
        mse: scalar.
    """
    
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_mse(y, tx, w)
    return w, loss

"----------------------------------------------------------------------------------------------------------------------"
"""                              Logistic Regression                                                                 """
"----------------------------------------------------------------------------------------------------------------------"

def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss).item() * (1 / y.shape[0])


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.shape[0])
    return grad


def logistic_gradient_descent(y,tx,initial_w,gamma,max_iters):

    ws = [initial_w]
    loss = 0
    w = initial_w

    for n_iter in range(max_iters):
            # compute a stochastic gradient and loss
            grad = calculate_gradient(y,tx,w)
            # update w through the stochastic gradient update
            w = w - gamma * grad     

    loss = calculate_loss(y, tx, w)
    return w, loss

def regd_logistic_regression(X, y, lr=0.01, max_iter=1000, fit_intercept=True, lambda_=0.1, verbose=False):
    if fit_intercept:
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
    theta = np.zeros(X.shape[1])
    
    for i in range(max_iter):
        z = np.dot(X, theta)
        h = 1 / (1 + np.exp(-z))
        gradient = np.dot(X.T, (h - y)) / y.size
        theta -= lr * gradient
        theta[1:] -= lr * lambda_ / y.size * theta[1:]  # Regularization term
        
        if(verbose and i % 10000 == 0):
            z = np.dot(X, theta)
            h = 1 / (1 + np.exp(-z))
            print(f'loss: {(-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()} \t')
    
    return theta