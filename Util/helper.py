import numpy as np


def train_test_spilt(X,Y,test_ratio=0.2,seed=None):
    # 将数据按照比例进行划分
    assert X.shape[0]==Y.shape[0],\
        "the size of x and y must be equal to each other"
    assert 0.0<test_ratio<1.0,\
        "the ratio must between （0,1）"
    if(seed):
        np.random.seed(seed)

    shuffled_indexes=np.random.permutation(len(X))

    test_size=int(len(X)*test_ratio)
    test_index=shuffled_indexes[:test_size]
    train_index=shuffled_indexes[test_size:]

    X_train=X[train_index]
    Y_train=Y[train_index]

    X_test=X[test_index]
    Y_test=Y[test_index]

    return X_train,Y_train,X_test,Y_test

    
        