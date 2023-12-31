import csv
import numpy as np
import matplotlib.pyplot as plt
import os as os

"----------------------------------------------------------------------------------------------------------------------"
"""                                                 Helper functions                                                 """
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


def cross(data_cleaned, pred, ratio):
    """
    Separate the data into a training and test set .

    :param data_cleaned: samples
    :param pred: labels
    :param ratio: the repartition of the sets (80%-20% usually)
    :return: the training and test sets
    """
    train_size = np.floor(data_cleaned.shape[0] * ratio).astype(int)

    tx_tr = data_cleaned[:train_size, :]
    y_tr = pred[:train_size, ]

    tx_te = data_cleaned[train_size:, :]
    y_te = pred[train_size:, ]

    return tx_tr, tx_te, y_tr, y_te


def build_model_data(data, pred):
    """Form (y,tX) to get regression data in matrix form."""
    y = pred
    x = data
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def pca(x_train):
    """
    Implementation of the PCA algorithm

    :param x_train: dataset to perform PCA onto
    :return: indices: the sorted indices of the features from the most influent one to the least one
             index: the number of features we can retain
    """
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


def best_threshold(y, tx, w):
    """
    Function to find the threshold that maximizes the F1-score.
    Additionally, this function plots the grid-search of the F1-score.

    :param y: labels
    :param tx: samples
    :param w: weights obtained after training
    :return: threshold that maximizes the F1-score
    """
    threshold = np.linspace(-1, 1, 100)

    best_f = 0
    best_thresh = -100
    f1_scores = []

    for el in threshold:
        pred_data = np.dot(tx, w)

        pred_data[pred_data > el] = 1
        pred_data[pred_data <= el] = -1

        tp = np.sum((pred_data == 1) & (y == 1))
        fp = np.sum((pred_data == 1) & (y == -1))
        fn = np.sum((pred_data == -1) & (y == 1))

        f_one = tp / (tp + 0.5 * (fn + fp))
        f1_scores.append(f_one)

        if f_one > best_f:
            best_f = f_one
            best_thresh = el

    plt.figure(figsize=(8, 6))
    plt.plot(threshold, f1_scores, label='F1-Score', color='b')
    plt.axvline(x=best_thresh, color='r', linestyle='--', label=f'Best Threshold (F1={best_f:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.title('Threshold vs. F1-Score')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_thresh


def gen_clean(raw_data, feat_cat, feat_con):
    """
    Standardize the data according to the type of the feature (categorical or continuous)

    :param raw_data: samples
    :param feat_cat: categorical features
    :param feat_con: continuous features
    :return: the standardized data
    """
    data = np.ones(raw_data.shape)

    for i in feat_con:
        d = standardize_clean(raw_data[:, i], False)
        data[:, i] = d

    for i in feat_cat:
        d = standardize_clean(raw_data[:, i], True)
        data[:, i] = d

    return data


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
        x[nan_indices] = 0

    if not categorical:
        x[nan_indices] = median_x

    x = x - np.mean(non_nan_values)
    std_x = np.std(x[~nan_indices])
    if std_x != 0:
        x = x / std_x

    return x


def replace(arr, old_values, new_values):
    """
    Replace all the old_values in arr by the corresponding in new_values

    :param arr: The array to perform the operation on
    :param old_values: Values to be replaced
    :param new_values: Replacement values
    :return: Array with the replaced values
    """
    result = arr.copy()
    for old_val, new_val in zip(old_values, new_values):
        result[result == old_val] = new_val

    return result


def preprocessing(x_train):
    """
    This function cleans the data for each features using the replace function defined above

    :param x_train: dataset to be cleaned
    :return: the cleaned dataset
    """
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
    x_train[:, 62] = list(map(into_pounds, (x_train[:, 62])))
    x_train[:, 63] = replace(x_train[:, 63], [7777, 9999], [np.nan, np.nan])
    x_train[:, 63] = list(map(into_inches, (x_train[:, 63])))
    x_train[:, 75] = replace(x_train[:, 75], [1, 2, 3, 4, 5, 6, 7, 8, 77, 99],
                             [15, 60, 135, 270, 1080, 2070, 3600, np.nan, np.nan, np.nan])
    x_train[:, 76] = replace(x_train[:, 76], [3, 7, 9], [0, np.nan, np.nan])
    x_train[:, 77] = replace(x_train[:, 77], [777, 888, 999], [np.nan, 0, np.nan])
    x_train[:, 77] = list(map(week_to_month, (x_train[:, 77])))
    array_5 = [78, 80, 88, 91, 98, 119]
    for i in array_5:
        x_train[:, i] = replace(x_train[:, i], [77, 99], [np.nan, np.nan])
    x_train[:, 79] = replace(x_train[:, 79], [77, 88, 99], [np.nan, 0, np.nan])
    array_6 = [81, 82, 83, 84, 85, 86]
    for i in array_6:
        x_train[:, i] = replace(x_train[:, i], [300, 555, 777, 999], [0, 0, np.nan, np.nan])
        x_train[:, i] = list(map(day_to_month, (x_train[:, i])))
    array_7 = [89, 90, 92, 93]
    for i in array_7:
        x_train[:, i] = replace(x_train[:, i], [777, 999], [0, 0, np.nan, np.nan])
    x_train[:, 89] = list(map(week_to_month, (x_train[:, 89])))
    x_train[:, 90] = list(map(hour_to_min, (x_train[:, 90])))
    x_train[:, 92] = list(map(hour_to_min, (x_train[:, 92])))
    array_8 = [94, 110, 111]
    for i in array_8:
        x_train[:, i] = replace(x_train[:, i], [777, 888, 999], [np.nan, 0, np.nan])
    x_train[:, 94] = replace(x_train[:, 94], [777, 888, 999], [np.nan, 0, np.nan])
    x_train[:, 94] = list(map(week_to_month, (x_train[:, 94])))
    x_train[:, 97] = replace(x_train[:, 97], [2, 3, 7, 9], [0.5, 0, np.nan, np.nan])
    x_train[:, 99] = replace(x_train[:, 99], [2, 3, 4, 5, 7, 8, 9], [0.75, 0.5, 0.25, 0, np.nan, np.nan, np.nan])
    x_train[:, 101] = replace(x_train[:, 101], [777777, 999999], [np.nan, np.nan])
    # x_train[:,101] = list(map(DateType,(x_train[:, 101])))
    x_train[:, 105] = replace(x_train[:, 105], [777777, 999999], [np.nan, np.nan])
    # x_train[:,105] = list(map(DateType,(x_train[:, 105])))
    x_train[:, 110] = list(map(day_to_year, (x_train[:, 110])))
    x_train[:, 111] = list(map(day_to_year, (x_train[:, 111])))
    x_train[:, 113] = replace(x_train[:, 113], [77, 88, 98, 99], [np.nan, 0, np.nan, np.nan])
    x_train[:, 114] = replace(x_train[:, 114], [77, 88, 99], [np.nan, 0, np.nan])
    x_train[:, 115] = replace(x_train[:, 114], [1, 2, 3, 4, 7, 8, 9], [15, 180, 540, 720, np.nan, 0, np.nan])
    nan79 = [120, 121, 123, 124, 125, 126, 129, 132, 136, 137, 138, 139, 140,
             141, 142, 144, 151, 154, 155, 156, 157, 158, 159, 160, 161, 162,
             163, 164, 165, 166, 169, 170, 171, 172, 173, 174, 175, 176, 177,
             178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
             191, 194, 196, 198, 199, 201, 202, 203, 204, 205, 214, 261]
    for i in nan79:
        x_train[:, i] = replace(x_train[:, i], [7, 9], [np.nan, np.nan])
    nan789 = [192, 193]
    for i in nan789:
        x_train[:, i] = replace(x_train[:, i], [7, 8, 9], [np.nan, np.nan, np.nan])
    nan7799 = [122, 130, 168, 224, 240]
    for i in nan7799:
        x_train[:, i] = replace(x_train[:, i], [77, 99], [np.nan, np.nan])
    x_train[:, 127] = replace(x_train[:, 127], [6, 7, 9], [np.nan, np.nan])
    x_train[:, 128] = replace(x_train[:, 128], [6, 7], [np.nan, np.nan])
    nan9 = [131, 153, 200, 223, 230, 231, 232, 233, 234, 235, 236, 241, 242, 243, 244, 255, 256, 257, 258, 259, 260,
            263,
            265, 278, 279, 298, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320]
    for i in nan9:
        x_train[:, i] = replace(x_train[:, i], [9], [np.nan])
    nan7 = [133, 134, 135, 146, 152]
    for i in nan7:
        x_train[:, i] = replace(x_train[:, i], [7], [np.nan])
    nan99900 = [264, 287, 288, 293, 294, 297]
    for i in nan99900:
        x_train[:, i] = replace(x_train[:, i], [99900], [np.nan])
    x_train[:, 143] = replace(x_train[:, 143], [777, 999], [np.nan, np.nan])
    x_train[:, 143] = list(map(convert_to_days, (x_train[:, 143])))
    x_train[:, 145] = list(map(asthme, (x_train[:, 145])))
    n088_98 = [147, 148]
    for i in n088_98:
        x_train[:, i] = replace(x_train[:, i], [88, 98], [0, np.nan])
    x_train[:, 149] = replace(x_train[:, 149], [88, 98, 99], [0, np.nan, np.nan])
    x_train[:, 150] = replace(x_train[:, 150], [777, 888, 999], [np.nan, 0, np.nan])
    x_train[:, 195] = replace(x_train[:, 195], [97, 98, 99], [np.nan, 0, np.nan])
    x_train[:, 197] = replace(x_train[:, 197], [97, 98, 99], [np.nan, 0, np.nan])
    nan088 = [206, 207, 208, 209, 210, 211, 212, 213]
    for i in nan088:
        x_train[:, i] = replace(x_train[:, i], [77, 88, 99], [np.nan, 0, np.nan])
    x_train[:, 225] = replace(x_train[:, 225], [7, 77, 99], [np.nan, np.nan, np.nan])
    x_train[:, 239] = replace(x_train[:, 239], [7, 77, 99], [np.nan, np.nan, np.nan])
    x_train[:, 246] = replace(x_train[:, 246], [14], [np.nan])
    x_train[:, 247] = replace(x_train[:, 247], [3], [np.nan])
    x_train[:, 262] = replace(x_train[:, 262], [900], [np.nan])

    return x_train


"----------------------------------------------------------------------------------------------------------------------"
"""                                  Conversion metrics functions for preprocessing                                  """
"----------------------------------------------------------------------------------------------------------------------"


def into_pounds(x):
    if x >= 9000:
        return int((x - 9000) * 2.20462)
    else:
        return x


def into_inches(x):
    if x < 9000:
        return np.floor(x / 100) * 12 + (x % 100)
    else:
        return (x - 9000) * 0.393701


def week_to_month(x):
    x_str = str(x)
    if x_str[0] == "1":
        return 4 * int(x_str[-4:-2])
    elif x_str[0] == "2":
        return int(x_str[-4:-2])
    else:
        return x


def day_to_month(x):
    x_str = str(x)
    if x_str[0] == "1":
        return 30 * int(x_str[-4:-2])
    elif x_str[0] == "2":
        return 4 * int(x_str[-4:-2])
    elif x_str[0] == "3":
        return int(x_str[-4:-2])
    else:
        return x


def day_to_year(x):
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


def hour_to_min(x):
    x_str = str(x)
    if len(x_str) == 4:
        return int(x_str[-4:-2])
    elif len(x_str) == 5:
        return int(x_str[0]) * 60 + int(x_str[-4:-2])
    else:
        return x


def convert_to_days(x):
    x = np.where((x >= 101) & (x < 200), x - 100, x)
    x = np.where((x >= 201) & (x < 300), x - 200, x)
    x = np.where((x >= 301) & (x < 400), x - 300, x)
    x = np.where((x >= 401) & (x < 500), x - 400, x)

    return x


def asthme(x):
    if x <= 97:
        return 1
    else:
        return 0


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
"""                                 Least squares regression using normal equations                                  """
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
    """Hyper-parameter tuning over cross-validation for parameters lambda and degree.
       Also plots the grid-search.

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

    >>> best_degree_selection(np.arange(2, 11), 4, np.logspace(-4, 0, 30))
    (7, 0.004520353656360241, 0.28957280566456634)
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    # losses for the graph
    loss_graph = []

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
            loss_graph.append(np.mean(rmse_te_temp))
        best_indice = np.argmin(rmse_te)
        best_lambdas.append(lambdas[best_indice])
        best_rmses.append(rmse_te[best_indice])

        print(f"Degree {degree} done !")

    ind_best_degree = np.argmin(best_rmses)
    best_degree = degrees[ind_best_degree]
    best_lambda = best_lambdas[ind_best_degree]
    best_rmse = best_rmses[ind_best_degree]

    D, L = np.meshgrid(degrees, lambdas)
    RMSE = np.array(loss_graph).reshape(D.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(np.log10(L), D, RMSE, cmap='viridis')
    ax.set_xlabel('log10(Lambda)')
    ax.set_ylabel('Degree')
    ax.set_zlabel('RMSE')
    ax.scatter([np.log10(best_lambda)], [best_degree], [best_rmse], color='red', s=100, label='Minimum RMSE')
    plt.title('RMSE for Different Degrees and Lambdas')
    plt.legend()
    plt.show()

    return best_degree, best_lambda, best_rmse


"----------------------------------------------------------------------------------------------------------------------"
"""                                               Logistic Regression                                                """
"----------------------------------------------------------------------------------------------------------------------"


def sigmoid(t):
    """
    The sigmoid loss function

    :return: the sigmoid loss
    """
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
        if iterable % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iterable, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    loss = calculate_loss(y, tx, w)

    return w, loss


def logistic_regression_demo(x_tr, y_tr, gammas, degrees, max_iters):
    """
    Hyper-parameter tuning for the paramters gamma and degree

    :param x_tr: samples
    :param y_tr: labels
    :param gammas: gamma values for grid-search
    :param degrees: degree values for grid-search
    :param max_iters: maximum number of iterations for GD of logistic regression
    :return: the best hyper-paramaters gamma and degree with the reespective loss
    """
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


def cat_sep(data, categorical_features):
    """
    One-hot encoding for the categorical features.

    [[1, 2] ----> [[1, 2, 1, 0, 1, 0]
    [3, 4]] ----> [3, 4, 0, 1, 0, 1]]

    :param data: dataset
    :param categorical_features: categorical data indices
    :return: the new dataset with the one-hot encoding for the categorical features
    """
    seperated_categories = data.copy()
    for feature in categorical_features:
        print(f"Feature : {feature}")
        col = data[:, feature]
        unique_values = np.unique(col)
        for val in unique_values:
            if val != 0:
                indices_val = np.where(col == val)
                indices_val_not = np.where(col != val)
                new_cat = col.copy()
                new_cat[indices_val] = 1
                new_cat[indices_val_not] = 0
                seperated_categories = np.c_[seperated_categories, new_cat]

    one_hot_data = np.delete(seperated_categories, categorical_features, axis=1)

    return one_hot_data


def gen_binary(raw_data, feat_cat, feat_con):
    """
    Standardize the data according to the type of the feature (categorical or continuous)
    for the special one-hot encoding experience.

    :param raw_data: samples
    :param feat_cat: categorical features
    :param feat_con: continuous features
    :return: the standardized data
    """
    data = np.ones(raw_data.shape)

    for i in feat_con:
        d = clean_binary(raw_data[:, i], False)
        data[:, i] = d

    for i in feat_cat:
        d = clean_binary(raw_data[:, i], True)
        data[:, i] = d

    return data


def clean_binary(x, categorical=True):
    """
    Replace NaN values in a feature with the median of the non-NaN values.
    Standardize and normalize only the continuous features.

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
        x[nan_indices] = 0

    if not categorical:
        x[nan_indices] = median_x
        x = x - np.mean(non_nan_values)
        std_x = np.std(x[~nan_indices])
        if std_x != 0:
            x = x / std_x

    return x
