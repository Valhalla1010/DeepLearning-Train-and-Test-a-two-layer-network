import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
from torch_gradient_computations import ComputeGradsWithTorch


# 1 this reads in the data from a CIFAR-10 bach file and
# returns the image and label data in separate files.
def LoadBatch(filename):
    """
    load a CIFAR-10 batch file
    :return:
        X contains the image pixel data d x n
        Y is K x n and contains the one-hot of the label for each image
        y is a vector of length n containing the label for each image.

    """
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    X = dict[b'data'].astype(np.float32) / 255.0
    y = np.array(dict[b'labels'])
    X = X.transpose()
    nn = X.shape[1]
    K = 10
    Y = np.zeros((K, nn), dtype=np.float32)
    arange = np.arange(nn)
    Y[y, arange] = 1

    return X, Y, y

# 2.Compute the mean and standard deviation vector
def preprocess(X_train, X_valid, X_test):
    d = X_train.shape[0] #
    mean_X = np.mean(X_train, axis=1).reshape(d, 1)
    std_X = np.std(X_train, axis=1).reshape(d, 1)

    # Normalize the datasets
    train_normalize = X_train - mean_X
    train_normalize = train_normalize / std_X
    valid_normalize = X_valid - mean_X
    valid_normalize = valid_normalize / std_X
    test_normalized = X_test - mean_X
    test_normalized = test_normalized / std_X

    return mean_X, std_X, train_normalize, valid_normalize, test_normalized


# 3. initialize the parameters of the network
"""
    initialize the parameters of the model W and b as you now know what
    size they should be. It is handy to store these network parameters in a
    dictionary. W has size K×d and b is K×1
"""
def initialize_parameters(d, K=10, seed=42):
    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    rng.bit_generator.state = BitGen(seed).state

    init_net = {}
    init_net['W'] = 0.01 * rng.standard_normal(size = (K, d))
    init_net['b'] = np.zeros((K, 1))

    return init_net


# 4-  forward pass (Softmax)
"""
     applies the network function, i.e. equations (1,
     2), to multiple images and returns the results
"""
def ApplyNetwork(X, network):
    """
     X = each clumn of X corresponds to an image, and size d x n
     network = is a dictioanry wirh keys W and b, that correspond to the parameters od the network
     p = each column of p contains the probability for each label of
     the image in the corresponding column of x, p size K X n.

    """
    W = network['W']
    b = network['b']
    s = W @ X + b # equation 1

    softmax = s - np.max(s, axis=0, keepdims=True) # equation 2
    exp_s = np.exp(softmax)
    p = exp_s / np.sum(exp_s, axis=0, keepdims=True)

    return p

# 5 cumoutes the loss function givan by equation 5
def ComputeLoss(P, y):
    """

    :param P:  each column of P is the probability of each class for the corresponding
                column of the data X and has size K×n
    :param y: y is (1×n) and corresponds to the ground truth label of each image
                whose predicted labels are contained in P
    :return:  L is a scalar corresponding to the mean cross-entropy loss of
                the network’s predictions relative to the ground truth labels.
    """
    n = P.shape[1]
    p = P[y, np.arange(n)]
    loss = -np.log(p)
    L = np.sum(loss) / n

    return L

# 6- computes the accuracy of the network’s
# predictions given by equation (4) on a set of data.
def CumputetAccuracy(P, y):
    """

    :param P:  each column of P is the probability of each class for the corresponding
                column of the data X and has size K×n
    :param y:   y is the vector of ground truth labels of length n.
    :return:
        acc is a scalar value containing the accuracy
    """
    p_label = np.argmax(P, axis=0)
    # compute with true labels
    k = (p_label == y)
    acc = np.mean(k)
    return acc

# 7  evaluates, for a mini-batch, the gradients of the
# cost function w.r.t. W and b, that is equations (10, 11)
def BackwardPass(X, Y, P, network, lam):
    n = X.shape[1]
    W = network['W']
    b = network['b']

    G = P - Y
    dW = (G @ X.T) / n + 2 * lam * W
    db = np.sum(G, axis=1, keepdims=True) / n
    grads = {}
    grads['W'] = dW
    grads['b'] = db
    return grads

# 8
def MiniBatchGD(X, Y, GDparams, network, lam):
    """
    :param X:  d x n
    :param Y:  K x n
    :param GDparams: a dictionary with keys:
        - 'n_batch': the number of images in each mini-batch
        - 'eta': the learning rate
        - 'n_epochs': the number of epochs to run the algorithm for
    :param network: a dictionary with keys W and b corresponding to the parameters of the network
    :param lam: the regularization parameter lambda
    :return:
        network: a dictionary with keys W and b corresponding to the parameters of the network after training.
    """
    trained_net = copy.deepcopy(network)
    n = X.shape[1]
    #  GDparams
    n_batch = GDparams['n_batch']
    eta = GDparams['eta']
    n_epochs = GDparams['n_epochs']
    rng = np.random.default_rng(42)
    for epoch in range(n_epochs):
        # shuffle the data at the beginning of each epoch
        perm = rng.permutation(n)
        X = X[:, perm]
        Y = Y[:, perm]
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            Xbatch = X[:, j_start:j_end]
            Ybatch = Y[:, j_start:j_end]

            P = ApplyNetwork(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch,P, trained_net, lam)

            # update parameters
            trained_net['W'] -= eta * grads['W']
            trained_net['b'] -= eta * grads['b']

    return trained_net
# 8 Implemented a function that trains the network using mini-batch gradient descent and plots the training and validation loss graphs.
def train_valid_graph(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, network, lam):
    n_epochs = GDparams['n_epochs']
    train_loss = []
    valid_loss = []

    inet_net = network
    for epoch in range(n_epochs):
        GDparams_epoch = GDparams.copy()
        GDparams_epoch['n_epochs'] = 1
        inet_net = MiniBatchGD(X_train, Y_train, GDparams_epoch, inet_net, lam)

        p_tarin = ApplyNetwork(X_train, inet_net)
        L_train = ComputeLoss(p_tarin, y_train)

        p_valid = ApplyNetwork(X_valid, inet_net)
        L_valid = ComputeLoss(p_valid, y_valid)

        train_loss.append(L_train)
        valid_loss.append(L_valid)

        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {L_train:.4f}, Valid Loss: {L_valid:.4f}")

    return inet_net, train_loss, valid_loss

# 8 plot the training and validation loss graphs
def PlotLoss(train_loss, valid_loss):
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.grid()
    plt.show()


# 8 visualize W matrix
def Visualize_W(trained_net, save=False):
    W = trained_net['W']
    Ws = W.transpose().reshape((32, 32, 3, 10), order='F')
    W_im = np.transpose(Ws, (1,0,2, 3))
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    for i in range(10):
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        ax = axs[i // 5, i % 5]
        ax.imshow(w_im_norm)
        ax.set_title(f"Class {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 1- load the data from the file data_batch_1 for training
def training():
    cifar_dir='Datasets/cifar-10-batches-py/'
    X_train, Y_train, y_train = LoadBatch(cifar_dir + 'data_batch_1')
    return X_train, Y_train, y_train

# 1- load the data from the file data_batch_2 and for validation
def validation():
    cifar_dir='Datasets/cifar-10-batches-py/'
    X_val, Y_val, y_val = LoadBatch(cifar_dir + 'data_batch_2')
    return X_val, Y_val, y_val
# 1- load the data from the file test_batch for testing
def test_batch():
    cifar_dir='Datasets/cifar-10-batches-py/'
    X_test, Y_test, y_test = LoadBatch(cifar_dir + 'test_batch')
    return X_test, Y_test, y_test

def display(X, Y, y, ni=5):
    fig, axs = plt.subplots(1, ni, figsize=(10, 5))
    for i in range(ni):
        img = X[:, i].reshape(3, 32, 32).transpose(1, 2, 0)
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f"Label: {y[i]}")
    plt.show()



def results(X, Y, y):
    print("X_train:", X.shape)
    print("Y_train:", Y.shape)
    print("y_train:", y.shape)
    for i in range(5):
        print("Label (y):", y[i])
        print("One-hot (Y column):", Y[:, i])
        print("Sum of one-hot column:", np.sum(Y[:, i]))


def main():
    # load the data
    # lebal for traning images X and Y
    # 1-read in and store the training, validation, and test data
    X, Y, y = training()
    X_val, Y_val, y_val = validation()
    X_test, Y_test, y_test = test_batch()
    # for training data
    #results(X, Y, y)
    # for test data
    #results(X_test, Y_test, y_test)
    # for validation data
    #results(X_val, Y_val, y_val)
    # show images and labels
    #display(X, Y, y)
    #display(X_test, Y_test, y_test)
    #display(X_val, Y_val, y_val)

    # 2: Pre-process compute the mean and standard deviation for training dara
    # and then normalize training, validation and test data
    mean_X, std_X, train_normalize, valid_normalize, test_normalized = preprocess(X, X_val, X_test)
    #print("Mean Vector ", mean_X[:5].flatten())
    #print("Std Vector ", std_X[:5].flatten())
    #print("Normalized Training ", train_normalize[:5].flatten())
    #print("Normalized Validation ", valid_normalize[:5].flatten())
    #print("Normalized Test", test_normalized[:5].flatten())

    # 3. initialize the parameters of the network
    init_net = initialize_parameters(X.shape[0], K=10, seed=42)
    #print("Initialized W :", init_net['W'][:5].flatten())
    #print("Initialized W shape :", init_net['W'].shape)
    #print("Initialized b :", init_net['b'][:5].flatten())
    #print("Initialized b shape :", init_net['b'].shape)
    #print("W std:", np.std(init_net['W']))


    #4.  Check the function runs on a subset of the training data
         #given a random initialization of the network’s parameters:
    P = ApplyNetwork(X[:, 0:100], init_net)
    #print("P:", P)
    #print("Sum of probabilities: \n", np.sum(P[:, 0:100], axis=0))


    # 5 compute the loss function
    #L = ComputeLoss(P, y[0:100])
    #print("Loss L:", L)

    # 6 compute the accuracy
    #acc = CumputetAccuracy(P, y[0:100])
    #print("Accuracy:", acc)

    #7 compute the gradients (mini-batch)
    #lam = 0
    #grads = BackwardPass(X[:, 0:100], Y[:, 0:100], P, init_net, lam)
    #print("Gradient dW:", grads['W'][:5].flatten(), "\n with shape:", grads['W'].shape)
    #print("Gradient db:", grads['b'][:5].flatten(), "\n with shape:", grads['b'].shape)

    # 7.
    d_small = 10
    n_small = 3
    lam = 0
    rng = np.random.default_rng(42)
    small_net = {}
    small_net['W'] = 0.01 * rng.standard_normal(size=(10, d_small))
    small_net['b'] = np.zeros((10, 1))
    x_small = X[0:d_small, 0:n_small]
    Y_small = Y[:, 0:n_small]
    y_small = y[0:n_small]
    p = ApplyNetwork(x_small, small_net)
    my_grads = BackwardPass(x_small, Y_small, p, small_net, lam)
    torch_grads = ComputeGradsWithTorch(x_small, y_small, small_net)
    print("Max abs diff W:", np.max(np.abs(my_grads['W'] - torch_grads['W'])))
    print("Max abs diff b:", np.max(np.abs(my_grads['b'] - torch_grads['b'])))


    # 8 min
    """GDparams = {
        'n_batch': 100,
        'eta': 0.01,
        'n_epochs': 40
    }
    lam = 0.1
    trained_net = MiniBatchGD(X, Y, GDparams, init_net, lam)
    p = ApplyNetwork(X, trained_net)
    train_data = CumputetAccuracy(p, y)
    print("Training accuracy:", train_data)
    # for graphs
    trained_net, train_loss, valid_loss = train_valid_graph(X, Y, y, X_val, Y_val, y_val, GDparams, init_net, lam)
    PlotLoss(train_loss, valid_loss)
    # for visualizing W matrix
    Visualize_W(trained_net)"""


if __name__ == '__main__':
    main()
