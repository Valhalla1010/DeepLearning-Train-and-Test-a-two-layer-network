import numpy as np
from Assignment1 import training, validation, test_batch, CumputetAccuracy, ComputeLoss, LoadBatch, PlotLoss, train_valid_graph
import copy
from torch_gradient_computations import ComputeGradsWithTorch
import matplotlib.pyplot as plt


#Exercise 1: Read in the data & initialize the parameters of the network
def initializeParameters(layer_size, seed):
    rng = np.random.default_rng(seed)

    L = len(layer_size) - 1 # (L-1)
    networkParams = {}
    networkParams['W'] = [None] * L
    networkParams['b'] = [None] * L
    for i in range(L):
        in_dim = layer_size[i]
        out_dim = layer_size[i + 1]
        rngStandM = rng.standard_normal((out_dim, in_dim))
        # weight
        networkParams['W'][i] = (1 / np.sqrt(in_dim)) * rngStandM
        # bias parameter
        networkParams['b'][i] = np.zeros((out_dim, 1))
    return networkParams

# Exercise 2: Compute the gradients for the network parameters
# Forward pass (2-layer + ReLU)
def ApplyNetwork2Layer(X, network):
    """
     X = each clumn of X corresponds to an image, and size d x n
     network = is a dictioanry wirh keys W and b, that correspond to the parameters od the network
     p = each column of p contains the probability for each label of
     the image in the corresponding column of x, p size K X n.

    """
    W1, b1 = network['W'][0],  network['b'][0]
    W2, b2 = network['W'][1],  network['b'][1]
    s1 = W1 @ X + b1 # equation 1
    h = np.maximum(0, s1) # ReLU activation function
    s = W2 @ h + b2
    softmax = s - np.max(s, axis=0, keepdims=True) # equation 2
    exp_s = np.exp(softmax)
    p = exp_s / np.sum(exp_s, axis=0, keepdims=True)

    fp_data = {
        'X': X,
        's1': s1,
        'h': h,
        's': s,
        'P': p
    }
    return p, fp_data
def ComputeCost(P, Y, network, lam):
    loss = -np.log(np.sum(Y * P, axis=0))
    loss = np.mean(loss)
    reg = 0
    for W in network['W']:
        reg += np.sum(W ** 2)
    cost = loss + lam * reg
    return cost


# Exercise 2: BackwardPass
def BackwardPass(X, Y, fp_data, network, lam):
    n = X.shape[1]

    w1, w2 = network['W'][0], network['W'][1]
    s1 = fp_data['s1']
    h = fp_data['h']
    P = fp_data['P']

    grads = {}
    grads['W'] = [None] * 2
    grads['b'] = [None] * 2

    # Output layer
    G = P - Y
    L2_1 = (G @ h.T) / n + 2 * lam * w2
    grads['W'][1] = L2_1
    grads['b'][1] = np.sum(G, axis=1, keepdims=True) / n

    # Backprop
    G = w2.T @ G
    G[s1 <= 0] = 0
    L2_0 = (G @ X.T) / n + 2 * lam * w1
    grads['W'][0] = L2_0
    grads['b'][0] = np.sum(G, axis=1, keepdims=True) / n

    return grads

# 2b
def MiniBatchGD2(X, Y, GDparams, network, lam):
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

            P, fp_data = ApplyNetwork2Layer(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch,fp_data, trained_net, lam)

            for i in range(len(trained_net['W'])):
                trained_net['W'][i] -= eta * grads['W'][i]
                trained_net['b'][i] -= eta * grads['b'][i]

    return trained_net


def costFunction(X, Y, y):
    d_small = 5
    n_small = 1000
    m = 50
    lam = 0  # L_2 regularization parameter
    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    seed = 42
    rng.bit_generator.state = BitGen(seed).state
    small_net = {}
    small_net['W'] = [None] * 2
    small_net['b'] = [None] * 2
    small_net['W'][0] = (1 / np.sqrt(d_small)) * rng.standard_normal(size=(m, d_small))
    small_net['b'][0] = np.zeros((m, 1))
    small_net['W'][1] = (1 / np.sqrt(m)) * rng.standard_normal(size=(10, m))
    small_net['b'][1] = np.zeros((10, 1))
    mean_X = np.mean(X, axis=1, keepdims=True)
    std_X = np.std(X, axis=1, keepdims=True)
    X = (X - mean_X) / std_X
    X_small = X[0:d_small, 0:n_small]
    Y_small = Y[:, 0:n_small]
    P, fp_data = ApplyNetwork2Layer(X_small, small_net)
    my_grads = BackwardPass(X_small, Y_small, fp_data, small_net, lam)
    torch_grads = ComputeGradsWithTorch(X_small, y[0:n_small], small_net)
    print("Max abs diff W1:", np.max(np.abs(my_grads['W'][0] - torch_grads['W'][0])))
    print("Max abs diff b1:", np.max(np.abs(my_grads['b'][0] - torch_grads['b'][0])))
    print("Max abs diff W2:", np.max(np.abs(my_grads['W'][1] - torch_grads['W'][1])))
    print("Max abs diff b2:", np.max(np.abs(my_grads['b'][1] - torch_grads['b'][1])))


# 2- for check my algorithm is ok.
#  overfit to the training data and get a very low loss on the training data after
# training for a sufficient number of epochs (∼200)
def overfittest(X, Y, y):
    mean_x = np.mean(X, axis=1, keepdims=True)
    std_x = np.std(X, axis=1, keepdims=True)
    X= (X - mean_x) / std_x
    n_small = 1000
    X_small = X[:, 0:n_small]
    Y_small = Y[:, 0:n_small]
    y_small = y[0:n_small]
    m = 50
    k = 10
    layer_size = [X.shape[0], m, k]
    init_net = initializeParameters(layer_size, seed=42)
    GDparams = {
        'n_batch': 10,
        'eta': 0.1,
        'n_epochs': 200
    }
    lam = 0
    trained_net = MiniBatchGD2(X_small, Y_small, GDparams, init_net, lam)
    P, _ = ApplyNetwork2Layer(X_small, trained_net)
    loss = ComputeLoss(P, y_small)
    print(f"Training loss on small dataset: {loss:.5f}")
    acc = CumputetAccuracy(P, y_small)
    print(f"Training accuracy on small dataset: {acc:.5f}")


# 3 Train your network with cyclical learning rates
def CyclicalLearningRate(eta_min, eta_max, n_s, t):
    cycle = np.floor(1 + t / (2 * n_s))
    x = np.abs(t / n_s - 2 * cycle + 1)
    eta = eta_min + (eta_max - eta_min) * np.maximum(0, (1 - x))
    return eta
# Training with cyclical learning rates to store loss, cost, accuracy
def MiniBatchGD_CLR(X, Y, y, X_valid, Y_valid, y_valid, GDparams, network, lam):
    trained_net = copy.deepcopy(network)
    n = X.shape[1]
    n_batch = GDparams['n_batch']
    n_epochs = GDparams['n_epochs']
    n_s = GDparams['n_s']
    eta_min = GDparams['eta_min']
    eta_max = GDparams['eta_max']
    t = 0
    rng = np.random.default_rng(42)
    train_loss, valid_loss = [],[]
    train_acc, valid_acc = [], []
    train_cost, valid_cost = [], []
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        X = X[:, perm]
        Y = Y[:, perm]
        y = y[perm]
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            Xbatch = X[:, j_start:j_end]
            Ybatch = Y[:, j_start:j_end]
            P, fp_data = ApplyNetwork2Layer(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch, fp_data, trained_net, lam)
            eta = CyclicalLearningRate(eta_min, eta_max, n_s, t)
            for i in range(len(trained_net['W'])):
                trained_net['W'][i] -= eta * grads['W'][i]
                trained_net['b'][i] -= eta * grads['b'][i]
            # time per cycle
            t += 1
            cycle = (t-1) // (2 * n_s) + 1
            if t % (2 * n_s // 10) == 0:
                P_train, _ = ApplyNetwork2Layer(X, trained_net)
                P_valid, _ = ApplyNetwork2Layer(X_valid, trained_net)
                train_loss.append(ComputeLoss(P_train, y))
                valid_loss.append(ComputeLoss(P_valid, y_valid))

                train_cost.append(ComputeCost(P_train, Y, trained_net, lam))
                valid_cost.append(ComputeCost(P_valid, Y_valid, trained_net, lam))

                train_acc.append(CumputetAccuracy(P_train, y))
                valid_acc.append(CumputetAccuracy(P_valid, y_valid))
                
                print(f"cycle={cycle},t={t}, eta={eta:.4f}, loss={train_loss[-1]:.4f}, cost={train_cost[-1]:.4f} ,acc={train_acc[-1]:.4f}")
    return trained_net, train_loss, valid_loss,train_cost,valid_cost ,train_acc, valid_acc

def plottrainvalid(train_loss, val_loss, train_acc, val_acc, train_cost, val_cost):
    plt.figure(figsize=(15, 5))
    # cost

    plt.subplot(1, 3, 1)
    plt.plot(train_cost, label='training', color='green')
    plt.plot(val_cost, label='validation', color='red')
    plt.xlabel("update step")
    plt.ylabel("cost")
    plt.title("Cost")
    plt.legend()
    plt.grid()

    #loss
    plt.subplot(1,3,2)
    plt.plot(train_loss, label='training', color='green')
    plt.plot(val_loss, label='validation', color='red')
    plt.xlabel("update step")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()
    plt.grid()

    #accuracy
    plt.subplot(1,3,3)
    plt.plot(train_acc, label='training', color='green')
    plt.plot(val_acc, label='validation', color='red')
    plt.xlabel("update step")
    plt.ylabel("accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()


# 4 Train your network for read (Coarse-to-fine random search)
# train + return best validation accuracy
def trainEvaluate(X, Y, y, X_valid, Y_valid, y_valid, lam, n_batch):
    d = [X.shape[0], 50, 10]
    init_net = initializeParameters(d, seed=42)
    n = X.shape[1]
    n_s = int(2 * np.floor(n / n_batch))
    GDparams = {
        'n_batch': n_batch,
        'n_epochs': 8,   # 4 one cycle, 3*4 = 12
        'n_s': n_s,
        'eta_min': 1e-5,
        'eta_max': 1e-1
    }
    trained_net, _, _, _, _, _, valid_acc = MiniBatchGD_CLR(X, Y, y, X_valid, Y_valid, y_valid, GDparams, init_net, lam)
    return max(valid_acc)

# 4, Coarse search
def coarseSearch(X, Y, y, X_valid, Y_valid, y_valid):
    l_min = -5
    l_max = -1
    l_values = np.linspace(l_min, l_max, 8)
    result = []
    rng = np.random.default_rng(42)
    for l in l_values:
        lam = 10 ** l
        acc = trainEvaluate(X, Y, y, X_valid, Y_valid, y_valid, lam, n_batch=100)
        print(f"lambda={lam:.5f}, validation accuracy={acc:.4f}")
        result.append((lam, acc))
    return result



# 4, fine search
def randomSearch(X, Y, y, X_valid, Y_valid, y_valid, best_lams):
    l_min = np.log10(min(best_lams))
    l_max = np.log10(max(best_lams))
    result = []
    rng = np.random.default_rng(42)
    for _ in range(8): # 8 random samples
        l = l_min + (l_max - l_min) * rng.random()
        lam = 10 ** l
        acc = trainEvaluate(X, Y, y, X_valid, Y_valid, y_valid, lam, n_batch=100)
        print(f"lambda={lam:.5f}, validation accuracy={acc:.4f}")
        result.append((lam, acc))
    return result

# 4 plot for coarse and fine search
def plotlamSearch(results, title):
    lams = [lam for lam, acc in results]
    accs = [acc for lam, acc in results]
    plt.figure(figsize=(8, 4))
    plt.scatter(lams, accs)
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Validation Accuracy')
    plt.title(title)
    plt.grid()
    plt.show()


# load all batches of data
def loadAllData():
    cifar_dir='Datasets/cifar-10-batches-py/'
    list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    X, Y, y = LoadBatch(cifar_dir + list[0])
    for batch in list[1:]:
        X_batch, Y_batch, y_batch = LoadBatch(cifar_dir + batch)
        X = np.hstack((X, X_batch))
        Y = np.hstack((Y, Y_batch))
        y = np.hstack((y, y_batch))
    return X, Y, y

# i)
def gradientcheck(X, Y, y):
    d_small = 20
    n_small = 100
    lam = 0.0
    X_small = X[0:d_small, 0:n_small]
    Y_small = Y[:, 0:n_small]
    y_small = y[0:n_small]
    m = 50
    rng = np.random.default_rng(42)
    small_net = {
        'W' : [(1 / np.sqrt(d_small)) * rng.standard_normal(size=(m, d_small)),
               (1 / np.sqrt(m)) * rng.standard_normal(size=(10, m))
        ],
        'b' : [np.zeros((m, 1)), np.zeros((10, 1))]
    }
    P, fp_data = ApplyNetwork2Layer(X_small, small_net)
    my_grads = BackwardPass(X_small, Y_small, fp_data, small_net, lam)
    torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net)
    diff_w1 = np.abs(my_grads['W'][0] - torch_grads['W'][0]).flatten()
    diff_w2 = np.abs(my_grads['W'][1] - torch_grads['W'][1]).flatten()
    diff_b1 = np.abs(my_grads['b'][0] - torch_grads['b'][0]).flatten()
    diff_b2 = np.abs(my_grads['b'][1] - torch_grads['b'][1]).flatten()
    print(f"Max abs diff W1: {np.max(diff_w1):.5e}")
    print(f"Max abs diff W2: {np.max(diff_w2):.5e}")
    print(f"Max abs diff b1: {np.max(diff_b1):.5e}")
    print(f"Max abs diff b2: {np.max(diff_b2):.5e}")

def main():

    X, Y, y = training() # data_bathch_1
    X_val, Y_val, y_val = validation() # file data_batch_2
    X_test, Y_test, y_test = test_batch() # file test_batch

    # E 1
    Xtrain = X.shape[0]
    #Xvalid = X_val.shape[0]
    #Xtest = X_test.shape[0]
    #m = 50
    #k = 10
    #d = [Xtrain, 50, 10]
    #init_net = initializeParameters(d, seed=42)
    #for i in range(len(init_net['W'])):
        #print(f"W[{i}] shape:", init_net['W'][i].shape)
        #print(f"b[{i}] shape:", init_net['b'][i].shape)
    

    # E 2 a) compute the network function
    #P, fp_data = ApplyNetwork2Layer(X[:, 0:100], init_net)
    #print("p first column:", P[:, 0])
    #print("Sum of probabilities:", np.sum(P[:, 0]))
    #print("\n Intermediate values:")
    #print("s_1 shape:", fp_data['s_1'].shape)
    #print("h shape:", fp_data['h'].shape)
    #print("s shape:", fp_data['s'].shape)
    #print("p shape:", fp_data['p'].shape)

    #grads = BackwardPass(X[:, 0:1000], Y[:, 0:1000], fp_data, init_net, lam=0)
    #print("\n Gradients:")
    #for i in range(len(grads['W'])):
     #   print(f"grads['W'][{i}] shape:", grads['W'][i].shape)
      #  print(f"grads['b'][{i}] shape:", grads['b'][i].shape)

    #2b) cost fuction
    #costFunction(X, Y, y)
    # 2b)
    #overfittest(X, Y, y)


    # 3
    #mean_x = np.mean(X, axis=1, keepdims=True)
    #std_x = np.std(X, axis=1, keepdims=True)
    #X = (X - mean_x) / std_x
    #X_val = (X_val - mean_x) / std_x
    #X_test = (X_test - mean_x) / std_x
    """GDparams = {
        'n_batch': 100,
        'n_epochs': 10, # 3 cycles, 2*n_s=a,  a/ n_batch = b, b*n_epochs. (48)
        'n_s': 500, # 500 to 1000
        'eta_min': 1e-5,
        'eta_max': 1e-1
    }
    lamm = 0.01
    trainednet, trainloss, validloss, traincost, validcost, trainacc, validacc = MiniBatchGD_CLR(
        X, Y, y, X_val, Y_val, y_val, GDparams, init_net, lamm
    )
    plottrainvalid(trainloss, validloss, trainacc, validacc, traincost, validcost)"""

    # 4 Loss/cross for more cycles:
    """GDparams = {
        'n_batch': 100,
        'n_epochs': 48, # 1 cycles, 2*n_s=a,  a/ n_batch = b, b * n_epochs.
        'n_s': 800, # 500 to 1000
        'eta_min': 1e-5,
        'eta_max': 1e-1
    }
    lam2= 0.01
    trainednet, trainloss, validloss, traincost, validcost, trainacc, validacc = MiniBatchGD_CLR(
        X, Y, y, X_val, Y_val, y_val, GDparams, init_net, lam2
    )
    P_test, _ = ApplyNetwork2Layer(X_test, trainednet)
    test_acc = CumputetAccuracy(P_test, y_test)
    print("Final test accuracy", test_acc)
    plottrainvalid(trainloss, validloss, trainacc, validacc, traincost, validcost)"""

    # load all data
    X_All, Y_All, y_All = loadAllData()
    # 4, coarse search
    """X_train = X_All[:, :-5000]
    Y_train = Y_All[:, :-5000]
    y_train = y_All[:-5000]
    X_valid = X_All[:, -5000:]
    Y_valid = Y_All[:, -5000:]
    y_valid = y_All[-5000:]
    # normalize
    mean_x = np.mean(X_train, axis=1, keepdims=True)
    std_x = np.std(X_train, axis=1, keepdims=True)
    X_train = (X_train - mean_x) / std_x
    X_valid = (X_valid - mean_x) / std_x

    # Coarse 
    print("##########  Coarse Search  ##########")
    coarse = coarseSearch(X_train, Y_train, y_train, X_valid, Y_valid, y_valid)
    plotlamSearch(coarse, "Coarse Search")
    #pick best 3 lambdas
    coarse.sort(key=lambda x: x[1], reverse= True)
    bestLam = coarse[:3]
    print("\n The 3 best performing network for COARSE SEARCH")
    for lam, acc in bestLam:
        print(f"lambda = {lam:.5f}, validation accuracy = {acc:.4f}")

    # fine search
    print("##########   fine search  ##########")
    best = [lam for lam, acc in coarse[:3]]
    fine = randomSearch(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, best)
    plotlamSearch(fine, "fine search")
    fine.sort(key=lambda x: x[1], reverse=True)
    best_lambda = fine[:3]
    print("\n  The 3 best performing network for fine Search ")
    for lam, acc in best_lambda:
        print(f"lambda={lam:.5f}, validation accuracy={acc:.4f}")"""

    #(v)
    xtrain = X_All[:, :-1000]
    Ytrain = Y_All[:, :-1000]
    ytrain = y_All[:-1000]
    xvalid = X_All[:, -1000:]
    Yvalid = Y_All[:, -1000:]
    yvalid = y_All[-1000:]
    mean_x = np.mean(xtrain, axis=1, keepdims=True)
    std_x = np.std(xtrain, axis=1, keepdims=True)
    xtrain = (xtrain - mean_x) / std_x
    xvalid = (xvalid - mean_x) / std_x
    X_test = (X_test - mean_x) / std_x
    n = xtrain.shape[1]
    n_batch = 100
    n_s = 2 * np.floor( n / n_batch)
    GDparams = {
        'n_batch': 100,
        'n_epochs': 12, #3*4=12
        'n_s': n_s,
        'eta_min': 1e-5,
        'eta_max': 1e-1
    }
    lam3 = 0.00001
    d = [X.shape[0], 50, 10]
    net = initializeParameters(d, seed=42)
    trainednet, trainloss, validloss, traincost, validcost, trainacc, validacc = MiniBatchGD_CLR(
        xtrain, Ytrain, ytrain, xvalid, Yvalid, yvalid, GDparams, net, lam3)
    plottrainvalid(trainloss, validloss, trainacc, validacc, traincost, validcost)

    P_test, _ = ApplyNetwork2Layer(X_test, trainednet)
    test_acc = CumputetAccuracy(P_test, y_test)
    print(f" Training accuracy =  {trainacc[-1]:.5f}")
    print(f"Final Test accuracy with best lambda = {test_acc:.5f}")
    print(f"Final Validation accuracy = {validacc[-1]:.5f}")


    # i)
    #gradientcheck(X, Y, y)
if __name__ == "__main__":
    main()