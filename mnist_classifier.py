import pickle
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lamb = 0.01


def relu(z):
    return np.maximum(0, z)


def one_hot_encoding(y):
    k = 10
    one_hot = np.zeros((len(y), k))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def relu_der(z):
    return np.where(z > 0, 1, 0)


def softmax(z):
    exp_x = np.exp(z)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def cost_function(Y, a_out, w1, w2):
    return (-1 * (np.sum(Y * np.log(a_out))) / len(Y)) + ((lamb / 2) * (np.sum(np.square(w1)) + np.sum(np.square(w2))))


def forward(X, w1, b1, w2, b2):
    z_hidden = np.dot(X, w1) + b1
    a_hidden = relu(z_hidden)
    z_out = np.dot(a_hidden, w2) + b2
    a_out = softmax(z_out)
    return z_hidden, a_hidden, z_out, a_out


def gradient(X, w1, w2, Y, z_hidden, a_hidden, a_out, total_records):
    dcost_dz_out = a_out - Y
    dzo_dw2 = a_hidden

    dcost_dw2 = (np.dot(dzo_dw2.T, dcost_dz_out) / total_records) + lamb * w2
    dcost_db2 = dcost_dz_out.sum(axis=0) / total_records

    dz_out_da_hidden = w2
    dcost_da_hidden = np.dot(dcost_dz_out, dz_out_da_hidden.T)

    da_hidden_dz_hidden = relu_der(z_hidden)
    dz_hidden_dw1 = X

    dcost_dw1 = (np.dot(dz_hidden_dw1.T, da_hidden_dz_hidden * dcost_da_hidden) / total_records) + lamb * w1
    dcost_db1 = (dcost_da_hidden * da_hidden_dz_hidden).sum(axis=0) / total_records

    return dcost_db2, dcost_dw2, dcost_db1, dcost_dw1


def create_mini_batch(indices):
    np.random.shuffle(indices)
    return np.split(indices, 100)


def plot(iterations, loss):
    plt.plot(iterations, loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epochs vs Loss')
    plt.show()


def train(X, Y, w1, b1, w2, b2):
    total_records = X.shape[0]
    learning_rate = 0.1
    z_hidden, a_hidden, z_out, a_out = forward(X, w1, b1, w2, b2)
    b2_gradient, w2_gradient, b1_gradient, w1_gradient = gradient(X, w1, w2, Y, z_hidden, a_hidden, a_out,
                                                                  total_records)
    b2 -= learning_rate * b2_gradient
    w2 -= learning_rate * w2_gradient
    b1 -= learning_rate * b1_gradient
    w1 -= learning_rate * w1_gradient
    z_hidden, a_hidden, z_out, a_out = forward(X, w1, b1, w2, b2)
    loss = cost_function(Y, a_out, w1, w2)
    correct_predictions = 0
    for i in range(len(a_out)):
        if np.argmax(a_out[i]) == np.argmax(Y[i]):
            correct_predictions += 1
    accuracy = correct_predictions / len(a_out)
    return accuracy, loss


def confusion_matrix(cm, X, one_hots, w1, w2, b1, b2):
    z_hidden, a_hidden, z_out, a_out = forward(X, w1, b1, w2, b2)
    for index in range(len(X)):
        if np.argmax(a_out[index]) == np.argmax(one_hots[index]):
            cm[np.argmax(a_out[index]), np.argmax(one_hots[index])] += 1
    return cm


def mini_batch(train_imgs, train__one_hot, val_imgs, val_one_hot, test_imgs, test_labels,
               test_one_hot, hidden_nodes):
    X = test_imgs

    output_nodes = len(np.unique(test_labels))
    features = X.shape[1]
    w1 = np.random.normal(scale=0.6, size=(features, hidden_nodes))
    b1 = np.random.random(hidden_nodes)
    w2 = np.random.normal(scale=0.6, size=(hidden_nodes, output_nodes))
    b2 = np.random.random(output_nodes)
    no_iterations = 20
    training_accuracy = []
    training_loss = []
    validation_accuracy = []
    validation_loss = []
    iterations = [x for x in range(no_iterations)]
    iter = 0
    for iter in iterations:
        batches = create_mini_batch(np.arange(len(train_imgs)))
        total_batch_accuracy = 0
        total_batch_loss = 0
        for batch in batches:
            batch_x = train_imgs.iloc[batch]
            batch_y = train__one_hot[batch]
            batch_accuracy, batch_loss = train(batch_x, batch_y, w1, b1, w2, b2)
            total_batch_accuracy += batch_accuracy
            total_batch_loss += batch_loss
        training_accuracy.append(total_batch_accuracy / len(batches))
        training_loss.append(total_batch_loss / len(batches))
        z_hidden, a_hidden, z_out, a_out = forward(val_imgs, w1, b1, w2, b2)

        current_loss = cost_function(val_one_hot, a_out, w1, w2)
        if len(validation_loss) > 0 and current_loss > validation_loss[-1]:
            break
        validation_loss.append(current_loss)
        correct_predictions = 0

        for i in range(len(a_out)):
            if np.argmax(a_out[i]) == np.argmax(val_one_hot[i]):
                correct_predictions += 1
        validation_accuracy.append(correct_predictions / len(a_out))
    z_hidden, a_hidden, z_out, a_out = forward(test_imgs, w1, b1, w2, b2)
    test_loss = cost_function(test_one_hot, a_out, w1, w2)
    class_accuracy = np.zeros(10)
    class_total = np.zeros(10)
    correct_predictions = 0
    for i in range(len(a_out)):
        if np.argmax(a_out[i]) == np.argmax(test_one_hot[i]):
            correct_predictions += 1
            class_accuracy[np.argmax(test_one_hot[i])] += 1
        class_total[np.argmax(test_one_hot[i])] += 1
    for index in range(10):
        class_accuracy[index] /= class_total[index]
    plt.plot(training_loss, label='Training loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.axvline(iter, 0, 10, color='g', label='Early Stopping')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.title('Loss values')
    plt.show()
    classes = np.arange(1, 11)
    plt.bar(classes, class_accuracy)
    plt.xticks(classes)
    plt.title('Class wise accuracy values')
    plt.xlabel('Class numbers')
    plt.ylabel('Accuracy values')
    plt.show()
    with open('weights.out', 'wb') as weight_file:
        pickle.dump([w1, w2, b1, b2], weight_file)
    print("Final Accuracy", correct_predictions / len(a_out))
    cm = np.zeros((10, 10), int)
    cm = confusion_matrix(cm, train_imgs, train__one_hot, w1, w2, b1, b2)
    cm = confusion_matrix(cm, val_imgs, val_one_hot, w1, w2, b1, b2)
    cm = confusion_matrix(cm, test_imgs, test_one_hot,w1, w2, b1, b2)
    print('Confusion matrix:', cm)


def process_mnist():
    train_df = pd.read_csv("mnist_train.csv", header=None)
    test_df = pd.read_csv("mnist_test.csv", header=None)

    train_imgs = train_df.iloc[:, 1:] / 255
    test_imgs = test_df.iloc[:, 1:] / 255
    train_labels = train_df[0]
    test_labels = test_df[0]

    val_indices = []

    for label in range(10):
        indices = train_labels.index[train_labels == label].tolist()
        val_indices.extend(random.sample(indices, int(len(indices) * 0.2)))
    val_imgs = train_imgs.iloc[val_indices]
    val_labels = train_labels.iloc[val_indices]
    train_imgs.drop(index=val_indices)
    train_labels.drop(index=val_indices)

    train__one_hot = one_hot_encoding(train_labels)
    val__one_hot = one_hot_encoding(val_labels)
    test_one_hot = one_hot_encoding(test_labels)

    return train_imgs, train_labels, train__one_hot, val_imgs, val_labels, val__one_hot, test_imgs, test_labels, test_one_hot


def main():
    np.random.seed(80)
    hidden_nodes = 32
    train_imgs, train_labels, train__one_hot, val_imgs, val_labels, val__one_hot, test_imgs, test_labels, test_one_hot = process_mnist()
    mini_batch(train_imgs, train__one_hot, val_imgs, val__one_hot, test_imgs, test_labels,
               test_one_hot, hidden_nodes)


if __name__ == '__main__':
    main()
    plt.show()
