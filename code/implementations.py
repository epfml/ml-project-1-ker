import csv
import numpy as np
import os as os


"----------------------------------------------------------------------------------------------------------------------"
"""                                         Helper functions                                                         """
"----------------------------------------------------------------------------------------------------------------------"


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids

def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
            

def gen_clean(raw_data):
    
    data = np.ones(raw_data.shape)
    stds = np.array([])
    for i in range(data.shape[1]):
        d, std = standardize_clean(raw_data[:, i])
        data[:, i] = d
        stds = np.append(stds, std)
        
    indices = np.where(stds != 0)
    data = data[:, indices]
    data = np.squeeze(data, axis = 1)
    
    data_cleaned = data[:, 10:]
        
    return data_cleaned, indices


def cross(data_cleaned, pred, ratio):
    
    train_size = np.floor(data_cleaned.shape[0] * ratio).astype(int)

    tx_tr = data_cleaned[:train_size, :]
    y_tr = pred[:train_size,]

    tx_te = data_cleaned[train_size:, :]
    y_te = pred[train_size:,]
    
    return tx_tr, tx_te, y_tr, y_te

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