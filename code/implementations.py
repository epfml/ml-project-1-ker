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
    
    data_cleaned = data[:, 9:]
        
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

def replace(arr, old_values, new_values):
    
    result = arr.copy()
    for old_val, new_val in zip(old_values, new_values):
        result[result == old_val] = new_val

    return result  


def pca(x_train):
    
    cov = np.cov(x_train.T)
    cov = np.round(cov, 2)
    
    eig_val, eig_vec = np.linalg.eig(cov)
    
    indices = np.arange(0,len(eig_val), 1)
    indices = ([x for _,x in sorted(zip(eig_val, indices))])[::-1]
    eig_val = eig_val[indices]
    eig_vec = eig_vec[:,indices]
    
    sum_eig_val = np.sum(eig_val)
    explained_variance = eig_val/ sum_eig_val
    cumulative_variance = np.cumsum(explained_variance)
    
    index = np.argmax(cumulative_variance > 0.95)
    
    return indices, index


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
    Returns:
        poly: numpy array of shape (N,d+1)
    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


"----------------------------------------------------------------------------------------------------------------------"
"""                                        Conversion metrics functions                                              """
"----------------------------------------------------------------------------------------------------------------------"


def IntoPounds(x):
    if x >= 9000 :
        return int((x - 9000) * 2.20462)
    else:
        return x 

    
def IntoInches(x):
    if x < 9000:          
        return np.floor(x/100)*12 + (x % 100)
    else: 
        return (x - 9000) * 0.393701

    
def WeekToMonth(x):
    x_str = str(x)
    if x_str[0] == "1":       
        return 4*int(x_str[-4:-2])
    elif x_str[0] == "2":
        return int(x_str[-4:-2])
    else :
        return x
    
    
def DayToMonth(x):
    x_str = str(x)
    if x_str[0] == "1":       
        return 30 *int(x_str[-4:-2])
    elif x_str[0] == "2":
        return 4*int(x_str[-4:-2])
    elif x_str[0] == "3":
        return int(x_str[-4:-2])
    else :
        return x
    
    
def DayToYear(x):
    x_str = str(x)
    if x_str[0] == "1":       
        return 365 *int(x_str[-4:-2])
    elif x_str[0] == "2":
        return 52*int(x_str[-4:-2])
    elif x_str[0] == "3":
        return 12 * int(x_str[-4:-2])
    elif x_str[0] == "4":
        return int(x_str[-4:-2])
    else:
        return x 

    
def HourToMinutes(x):
    x_str = str(x)
    if len(x_str) == 4 :
        return int(x_str[-4:-2])
    elif len(x_str) == 5 :   
        return int(x_str[0])*60 + int(x_str[-4:-2])
    else: 
        return x

    
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
"""                                      Ridge regression using normal equations                                     """
"----------------------------------------------------------------------------------------------------------------------"


def ridge_regression(y, tx, lambda_):
    """Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features
        lambda_: scalar.
       Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    
    return np.linalg.solve(a, b)


def ridge_regression_demo(x_tr, x_te, y_tr, y_te,lambdas,degrees) : 
    
    best_lambdas = []
    best_rmses = []
    
    for degree in degrees:
        rmse_te = []
        
        tx_te = build_poly(x_te, degree)
        tx_tr = build_poly(x_tr, degree)
        
        
        for lam in lambdas:
            rmse_te_tmp = []
            weight = ridge_regression(y_tr, tx_tr, lam)
            rmse_te_tmp.append(np.sqrt(2 * compute_loss_mse(y_te, tx_te, weight)))
                               
        rmse_te.append(np.mean(rmse_te_tmp))
            
        ind_lambda_opt = np.argmin(rmse_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_rmses.append(rmse_te[ind_lambda_opt])
        
    ind_best_degree = np.argmin(best_rmses)
    best_degree = degrees[ind_best_degree]
    best_lambda = best_lambdas[ind_best_degree]
    best_rmse = best_rmses[ind_best_degree]
                               
    return best_degree, best_lambda, best_rmse


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