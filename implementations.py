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


def gen_clean(raw_data, feat_cat, feat_con):
    data = np.ones(raw_data.shape)

    for i in feat_con:
        d, std = standardize_clean(raw_data[:, i], False)
        data[:, i] = d

    for i in feat_cat:
        d, std = standardize_clean(raw_data[:, i], True)
        data[:, i] = d

    return data


def cross(data_cleaned, pred, ratio):
    train_size = np.floor(data_cleaned.shape[0] * ratio).astype(int)

    tx_tr = data_cleaned[:train_size, :]
    y_tr = pred[:train_size, ]

    tx_te = data_cleaned[train_size:, :]
    y_te = pred[train_size:, ]

    return tx_tr, tx_te, y_tr, y_te


def standardize_clean(x, categorical=True):
    """
    Replace NaN values in a feature with the median of the non-NaN values.

    Args:
        :param x: feature to be standardized
        :param categorical: boolean representing if it is a categorical feature or not

    Returns:
        numpy.ndarray: 1D array with NaN values replaced by the median.
    """
    nan_indices = np.isnan(x)
    non_nan_values = x[~nan_indices]  # Get non-NaN values
    median_x = np.median(non_nan_values)

    if categorical:
        x[nan_indices] = -1

    if not categorical:
        x[nan_indices] = median_x

    x = x - median_x
    std_x = np.std(x[~nan_indices])
    if std_x != 0:
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

    indices = np.arange(0, len(eig_val), 1)
    indices = ([x for _, x in sorted(zip(eig_val, indices))])[::-1]
    eig_val = eig_val[indices]

    sum_eig_val = np.sum(eig_val)
    explained_variance = eig_val / sum_eig_val
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
    x_poly = np.ones((len(x), 1))
    for i in range(1, degree + 1):
        x_pow = x ** i
        x_poly = np.c_[x_poly, x_pow]

    return x_poly


"----------------------------------------------------------------------------------------------------------------------"
"""                                        Conversion metrics functions                                              """
"----------------------------------------------------------------------------------------------------------------------"


def IntoPounds(x):
    if x >= 9000:
        return int((x - 9000) * 2.20462)
    else:
        return x


def IntoInches(x):
    if x < 9000:
        return np.floor(x / 100) * 12 + (x % 100)
    else:
        return (x - 9000) * 0.393701


def WeekToMonth(x):
    x_str = str(x)
    if x_str[0] == "1":
        return 4 * int(x_str[-4:-2])
    elif x_str[0] == "2":
        return int(x_str[-4:-2])
    else:
        return x


def DayToMonth(x):
    x_str = str(x)
    if x_str[0] == "1":
        return 30 * int(x_str[-4:-2])
    elif x_str[0] == "2":
        return 4 * int(x_str[-4:-2])
    elif x_str[0] == "3":
        return int(x_str[-4:-2])
    else:
        return x


def DayToYear(x):
    x_str = str(x)
    if x_str[0] == "1":
        return 365 * int(x_str[-4:-2])
    elif x_str[0] == "2":
        return 52 * int(x_str[-4:-2])
    elif x_str[0] == "3":
        return 12 * int(x_str[-4:-2])
    elif x_str[0] == "4":
        return int(x_str[-4:-2])
    else:
        return x


def HourToMinutes(x):
    x_str = str(x)
    if len(x_str) == 4:
        return int(x_str[-4:-2])
    elif len(x_str) == 5:
        return int(x_str[0]) * 60 + int(x_str[-4:-2])
    else:
        return x


"----------------------------------------------------------------------------------------------------------------------"
"""                                     Linear regression using gradient descent                                     """
"----------------------------------------------------------------------------------------------------------------------"


def compute_loss_mse(y, tx, w):
    """compute the loss by mse.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: weights, numpy array of shape(D,), D is the number of features.
    Returns:
        mse: scalar corresponding to the mse with factor (1 / 2 n) in front of the sum
    >>> compute_loss_mse(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([0.03947092, 0.00319628]))
    0.006417022764962313
    """

    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


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
        tx: numpy array of shape=(N, M)
        initial_w: numpy array of shape=(M, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (M, ),
            for each iteration of GD
    """

    w = initial_w  # Initialize model parameters

    for n_iter in range(max_iters):
        loss = compute_loss_mse(y, tx, w)  # Compute the loss
        grad = compute_gradient_mse(y, tx, w)  # Compute the gradient
        w = w - gamma * grad  # Update the model parameters
        print(loss)

    loss = compute_loss_mse(y, tx, w)

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
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (M, )
        , for each iteration of SGD
    """

    w = initial_w
    loss = compute_loss_mse(y, tx, w)
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
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    a = tx.T.dot(tx) + 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_mse(y, tx, w)

    return w, loss


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)

    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]

    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    w, loss = ridge_regression(y_tr, tx_tr, lambda_)

    loss_tr = np.sqrt(2 * compute_loss_mse(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_loss_mse(y_te, tx_te, w))
    return loss_tr, loss_te


def best_degree_selection(y, x, degrees, k_fold, lambdas, seed=1):
    """cross validation over regularisation parameter lambda and degree.

    Args:
        y: labels of shape (n, )
        x: samples of shape (n, c)
        degrees: shape = (d,), where d is the number of degrees to test
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
        seed: random seed
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)

    >>> best_degree_selection(np.arange(2,11), 4, np.logspace(-4, 0, 30))
    (7, 0.004520353656360241, 0.28957280566456634)
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    best_lambdas = []
    best_rmses = []
    for degree in degrees:
        rmse_te = []
        for lambda_ in lambdas:
            rmse_te_temp = []
            for k in range(k_fold):
                _, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree)
                rmse_te_temp.append(loss_te)
            rmse_te.append(np.mean(rmse_te_temp))
        best_indice = np.argmin(rmse_te)
        best_lambdas.append(lambdas[best_indice])
        best_rmses.append(rmse_te[best_indice])

        print(f"Degree {degree} done !")

    ind_best_degree = np.argmin(best_rmses)
    best_degree = degrees[ind_best_degree]
    best_lambda = best_lambdas[best_degree]
    best_rmse = best_rmses[best_degree]

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

    >>> y_loss = np.c_[[0., 1.]]
    >>> tx_loss = np.arange(4).reshape(2, 2)
    >>> w_loss = np.c_[[2., 3.]]
    >>> round(calculate_loss(y_loss, tx_loss, w_loss), 8)
    1.52429481
    """

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    sig = sigmoid(tx.dot(w))
    matrix = y * np.log(sig) + (1 - y) * np.log(1 - sig)
    loss = -(1 / len(y)) * np.sum(matrix)

    return loss


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y_grad = np.c_[[0., 1.]]
    >>> tx_grad = np.arange(6).reshape(2, 3)
    >>> w_grad = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y_grad, tx_grad, w_grad)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """

    sig = sigmoid(tx.dot(w))
    grad = tx.T.dot(sig - y) * (1 / y.shape[0])

    return grad


def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a hessian matrix of shape=(D, D)

    >>> y_te = np.c_[[0., 1.]]
    >>> tx_te = np.arange(6).reshape(2, 3)
    >>> w_te = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_hessian(y_te, tx_te, w_te)
    array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]])
    """

    sig = sigmoid(tx.dot(w)).reshape(-1, 1)
    diag = np.diag(sig.T[0])
    s = np.multiply(diag, (1 - diag))
    hessian = (1 / y.shape[0]) * tx.T.dot(s.dot(tx))

    return hessian


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Do gradient descent with Newton's method.
    Return optimal w and loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1)
        max_iters: # of iterations
        gamma: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """

    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression
    for iterable in range(max_iters):
        # get loss and update w.

        loss = calculate_loss(y, tx, w)
        grad = calculate_gradient(y, tx, w)

        w = w - gamma * grad

        # log info
        # if iter % 100 == 0:
        #    print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    loss = calculate_loss(y, tx, w)

    return w, loss


def logistic_regression_demo(x_tr, y_tr, gammas, degrees, max_iters):
    best_gammas = []
    best_losses = []

    for degree in degrees:
        rmse_te = []

        tx_tr = build_poly(x_tr, degree)

        initial_w = np.zeros(tx_tr.shape[1])

        loss_te_tmp = []
        for gamma in gammas:
            weight, loss = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
            loss_te_tmp.append(loss)

        rmse_te.append(np.mean(loss_te_tmp))

        ind_lambda_opt = np.argmin(rmse_te)
        best_gammas.append(gammas[ind_lambda_opt])
        best_losses.append(rmse_te[ind_lambda_opt])

        print("Degree :", degree, " done")

    ind_best_degree = np.argmin(best_losses)
    best_degree = degrees[ind_best_degree]
    best_gamma = best_gammas[ind_best_degree]
    best_loss = best_losses[ind_best_degree]

    return best_degree, best_gamma, best_loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Do gradient descent, using the penalized logistic regression.
    Return the optimal w and loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1)
        max_iters: # of iterations
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """

    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression
    for iterable in range(max_iters):
        # get loss and update w.

        loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad

        # log info
        if iterable % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iterable, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    loss = calculate_loss(y, tx, w)
    print("loss={l}".format(l=loss))

    return w, loss


def preprocessing(x_train):
    x_train[:, 6] = replace(x_train[:, 6], [1100, 1200], [1, 0])
    x_train[:, 13] = replace(x_train[:, 13], [0, 1], [1, 2])
    x_train[:, 24] = replace(x_train[:, 24], [1, 2, 7, 9], [0, 1, np.nan, np.nan])
    x_train[:, 25] = replace(x_train[:, 25], [77, 99], [np.nan, np.nan])
    x_train[:, 26] = replace(x_train[:, 26], [2, 3, 4, 5, 7, 9], [0.75, 0.5, 0.25, 0, np.nan, np.nan])

    array_1 = [27, 28, 29]

    for i in array_1:
        x_train[:, i] = replace(x_train[:, i], [88, 77, 99], [np.nan, np.nan, np.nan])

    x_train[:, 31] = replace(x_train[:, 31], [3, 7, 9], [0, np.nan, np.nan])

    array_2 = [30, 32, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 53, 54, 55, 56, 57, 61, 64,
               65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 87, 95, 96, 100, 103, 104, 107, 108, 116, 117, 118]

    for i in array_2:
        x_train[:, i] = replace(x_train[:, i], [7, 9], [np.nan, np.nan])

    x_train[:, 33] = replace(x_train[:, 33], [1, 2, 3, 4, 7, 8, 9], [6, 18, 42, 60, np.nan, 120, np.nan])
    x_train[:, 37] = replace(x_train[:, 37], [1, 2, 3, 4, 7, 9], [6, 18, 42, 60, np.nan, np.nan])
    x_train[:, 49] = replace(x_train[:, 49], [98, 99], [np.nan, np.nan])

    array_3 = [51, 52, 58]

    for i in array_3:
        x_train[:, i] = replace(x_train[:, i], [9], [np.nan])

    x_train[:, 59] = replace(x_train[:, 59], [88, 99], [0, np.nan])
    x_train[:, 60] = replace(x_train[:, 60], [1, 2, 3, 4, 5, 6, 7, 8, 77, 99],
                             [5, 12.5, 17.5, 22.5, 30, 42.5, 62.5, 75, np.nan, np.nan])

    x_train[:, 62] = replace(x_train[:, 62], [7777, 9999], [np.nan, np.nan])
    x_train[:, 62] = list(map(IntoPounds, (x_train[:, 62])))

    x_train[:, 63] = replace(x_train[:, 63], [7777, 9999], [np.nan, np.nan])
    x_train[:, 63] = list(map(IntoInches, (x_train[:, 63])))

    x_train[:, 75] = replace(x_train[:, 75], [1, 2, 3, 4, 5, 6, 7, 8, 77, 99],
                             [15, 60, 135, 270, 1080, 2070, 3600, np.nan, np.nan, np.nan])
    x_train[:, 76] = replace(x_train[:, 76], [3, 7, 9], [0, np.nan, np.nan])
    x_train[:, 77] = replace(x_train[:, 77], [777, 888, 999], [np.nan, 0, np.nan])
    x_train[:, 77] = list(map(WeekToMonth, (x_train[:, 77])))

    array_5 = [78, 80, 88, 91, 98, 119]

    for i in array_5:
        x_train[:, i] = replace(x_train[:, i], [77, 99], [np.nan, np.nan])

    x_train[:, 79] = replace(x_train[:, 79], [77, 88, 99], [np.nan, 0, np.nan])

    array_6 = [81, 82, 83, 84, 85, 86]

    for i in array_6:
        x_train[:, i] = replace(x_train[:, i], [300, 555, 777, 999], [0, 0, np.nan, np.nan])
        x_train[:, i] = list(map(DayToMonth, (x_train[:, i])))

    array_7 = [89, 90, 92, 93]

    for i in array_7:
        x_train[:, i] = replace(x_train[:, i], [777, 999], [0, 0, np.nan, np.nan])

    x_train[:, 89] = list(map(WeekToMonth, (x_train[:, 89])))
    x_train[:, 90] = list(map(HourToMinutes, (x_train[:, 90])))
    x_train[:, 92] = list(map(HourToMinutes, (x_train[:, 92])))

    array_8 = [94, 110, 111]

    for i in array_8:
        x_train[:, i] = replace(x_train[:, i], [777, 888, 999], [np.nan, 0, np.nan])

    x_train[:, 94] = replace(x_train[:, 94], [777, 888, 999], [np.nan, 0, np.nan])
    x_train[:, 94] = list(map(WeekToMonth, (x_train[:, 94])))
    x_train[:, 97] = replace(x_train[:, 97], [2, 3, 7, 9], [0.5, 0, np.nan, np.nan])
    x_train[:, 99] = replace(x_train[:, 99], [2, 3, 4, 5, 7, 8, 9], [0.75, 0.5, 0.25, 0, np.nan, np.nan, np.nan])
    x_train[:, 101] = replace(x_train[:, 101], [777777, 999999], [np.nan, np.nan])

    # x_train[:,101] = list(map(DateType,(x_train[:, 101])))

    x_train[:, 105] = replace(x_train[:, 105], [777777, 999999], [np.nan, np.nan])
    # x_train[:,105] = list(map(DateType,(x_train[:, 105])))

    x_train[:, 110] = list(map(DayToYear, (x_train[:, 110])))
    x_train[:, 111] = list(map(DayToYear, (x_train[:, 111])))

    x_train[:, 113] = replace(x_train[:, 113], [77, 88, 98, 99], [np.nan, 0, np.nan, np.nan])
    x_train[:, 114] = replace(x_train[:, 114], [77, 88, 99], [np.nan, 0, np.nan])
    x_train[:, 115] = replace(x_train[:, 114], [1, 2, 3, 4, 7, 8, 9], [15, 180, 540, 720, np.nan, 0, np.nan])

    x_train[:, 240] = replace(x_train[:, 240], [np.nan, 77, 99], [-1, -1, -1])
    x_train[:, 246] = replace(x_train[:, 246], [np.nan, 14], [-1, -1])
    x_train[:, 247] = replace(x_train[:, 247], [np.nan, 3], [-1, -1])
    x_train[:, 252] = replace(x_train[:, 252], [np.nan, 99999], [np.nan, np.nan])
    x_train[:, 261] = replace(x_train[:, 261], [np.nan, 7, 9], [-1, -1, -1])
    x_train[:, 262] = replace(x_train[:, 262], [np.nan, 900], [-1, -1])
    x_train[:, 298] = replace(x_train[:, 298], [np.nan, 9], [1, 1])

    rep_one = [241, 242, 243, 244, 255, 256, 257, 258, 259, 260, 263, 265, 278, 279, 284,
               305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320]

    for i in rep_one:
        x_train[:, i] = replace(x_train[:, i], [np.nan, 9], [-1, -1])

    rep_two = [245, 249, 254, 289, 290, 291, 292]

    for i in rep_two:
        x_train[:, i] = replace(x_train[:, i], [np.nan], [-1])

    x_train[:, 264] = replace(x_train[:, 264], [np.nan, 99900], [-1, -1])
    x_train[:, 287] = replace(x_train[:, 287], [np.nan, 99900], [-1, -1])
    x_train[:, 288] = replace(x_train[:, 288], [np.nan, 99900], [-1, -1])

    x_train[:, 272] = replace(x_train[:, 272], [np.nan], [0])
    x_train[:, 273] = replace(x_train[:, 273], [np.nan], [0])

    x_train[:, 274] = replace(x_train[:, 274], [np.nan], [1])
    x_train[:, 275] = replace(x_train[:, 275], [np.nan], [1])
    x_train[:, 280] = replace(x_train[:, 280], [np.nan], [1])
    x_train[:, 281] = replace(x_train[:, 281], [np.nan], [1])

    x_train[:, 282] = replace(x_train[:, 282], [np.nan], [2])
    x_train[:, 283] = replace(x_train[:, 283], [np.nan], [2])

    x_train[:, 293] = replace(x_train[:, 293], [np.nan, 99000], [np.nan, np.nan])
    x_train[:, 294] = replace(x_train[:, 294], [np.nan, 99000], [np.nan, np.nan])
    x_train[:, 297] = replace(x_train[:, 297], [np.nan, 99000], [np.nan, np.nan])

    return x_train


def cat_sep(data, categorical_features):

    seperated_categories = data
    for feature in categorical_features:
        col = data[:, feature]
        unique_values = np.unique(col)
        for val in unique_values:
            if val != 0:
                indices_val = np.where(seperated_categories == val)
                new_cat = col
                new_cat[indices_val:, :] = val
                new_cat[~indices_val:, :] = 0
                seperated_categories = np.c_[seperated_categories, new_cat]

    return seperated_categories
