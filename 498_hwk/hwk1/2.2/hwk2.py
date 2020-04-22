import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

traindata = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None).values
positive = np.where(traindata == ' <=50k')
negative = np.where(traindata == ' >50k')
traindata[positive] = -1
traindata[negative] = 1
traindata = traindata[:, [0,2,4,10,11,12,-1]]  # only use numerical features and the last column(label)
mean = np.mean(traindata[:, 0:-1], axis=0)
std = np.std(traindata[:, 0:-1].astype(int), axis=0)
split = int(0.9 * traindata.shape[0])

# data unit normalization
traindata[:, 0:-1] = np.divide(np.subtract(traindata[:, 0:-1], mean), std)

lambdas = np.array([0.001, 0.01, 0.1, 1])
train_acc = np.ones((4, 10))
test_acc = np.ones((4, 1))

# train and evaluate model for different lambdas
for i in range(4):

    best_acc = 0
    for epo in range(50):  # train the model for 50 epochs

        # randomly split the data set into 90% training and 10% testing
        rand_idx = np.arange(traindata.shape[0])
        np.random.shuffle(rand_idx)
        train = traindata[rand_idx[0:split]]
        test = traindata[rand_idx[split:]]

        # select a small portion of training data as held out and the rest to be epoch dataset
        held_out = train[0:50]
        epoch = train[50:]
        steps = 300
        batch_size = int(epoch.shape[0] / steps)  # batch_size = epoch_size / step
        held_out_acc = []

        # initialize a and b
        a = np.ones((1, 6))
        b = 1

        for s in range(steps):

            step_length = 1 / (0.01 * s + 20)  # variant step length
            batch = epoch[s * batch_size: (s + 1) * batch_size]
            boundary = np.dot(batch[:, -1].T, (np.dot(batch[:, 0:-1], a.T) + b))  # y*(a*x+b)
            if boundary >= 1:
                a = a - step_length * lambdas[i] * a
            else:
                a = a - step_length * (lambdas[i] * a - batch[-1, -1] * batch[-1, 0:-1])
                b = b + step_length * batch[-1, -1]

            if s % 30 == 0:  # examine the model accuracy on held out data for every 30 steps
                held_out_pred = np.sign(np.dot(held_out[:, 0:-1], a.T) + b)
                held_out_err = np.where(held_out_pred.T != held_out[:, -1])[0].shape[0]
                held_out_acc = 1 - held_out_err / held_out.shape[0]
                mark = int(s / 30)
                train_acc[i, mark] = held_out_acc

        pred = np.sign(np.dot(test[:, 0:-1], a.T) + b)
        err = np.where(pred.T != test[:, -1])[0].shape[0]
        acc = 1 - err / test.shape[0]
        if acc > best_acc:
            best_acc = acc

    test_acc[i] = best_acc

# plot accuracy plots for different lambda values

x = np.linspace(0, 270, num=10)
y1 = train_acc
y2 = test_acc
for i in range(4):
    plt.figure()
    plt.plot(x, y1[i], marker='o', mec='r', mfc='w')
    plt.title('training accuracy for lambda = ' + str(lambdas[i]))
    plt.xlabel('steps')
    plt.ylabel('accuracy')
    plt.show()

plt.figure()
plt.plot(lambdas, y2, marker='o', mec='r', mfc='w')
plt.title('testing accuracy for different lambda')
plt.xlabel('lambda')
plt.ylabel('accuracy')
plt.show()
