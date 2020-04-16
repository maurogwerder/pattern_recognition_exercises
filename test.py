#########################
# Exercise 2b: MLP      #
# Author: Mauro Gwerder #
#########################
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
# https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2
# https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
# https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
# https://analyticsindiamag.com/a-beginners-guide-to-scikit-learns-mlpclassifier/
# https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3

# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/

# http://www.deeplearningbook.org/
import numpy as np
import matplotlib.pyplot as plt
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import csv
import pandas as pd


#https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
def load_data():
    with open("mnist_train.csv", "r") as file:
        reader = csv.reader(file)
        data = list(reader)  # converts csv.reader object into list of lists, each list being one line from the file
        matrix = np.array(data[1:], dtype=int)  # converts list of lists into an array of type int
        return matrix


def classifier():
    classifier = MLPClassifier(hidden_layer_sizes=(100, ),
                               alpha=0.0001,
                               verbose=True,
                               activation='logistic',
                               solver='sgd',
                               max_iter=20,
                               random_state=1,
                               learning_rate='constant',
                               learning_rate_init=0.001,
                               # warm_start=True,
                               batch_size=50,
                               # max_iter=300,
                               )
    return classifier


def curve(data, classifier):
    X = data[:10000, 1:]
    y = data[:10000, 0]
    labels = np.unique(y)
    lb = LabelBinarizer()
    lb.fit(labels)
    y = lb.transform(y)
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    # https://scikit-learn.org/stable/modules/learning_curve.html
    train_sizes, train_scores, valid_scores, fit_times, _ = learning_curve(classifier,
                                                                           X, y,
                                                                           train_sizes=np.linspace(.1, 1.0, 5),
                                                                           cv=4,
                                                                           return_times=True)
    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    plt.plot(train_sizes, train_mean, label="Training set")
    plt.plot(train_sizes, valid_mean, label="Validation set")
    plt.legend()
    plt.title("Learning Curve: Accuracy")
    plt.ylabel("Score")
    plt.xlabel("Iterations")
    plt.show()


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


def training(data, classifier):
    scaler = StandardScaler()
    splits = KFold(4)
    acc_list = []

    # appends model by one iteration.
    for training_set, validation_set in splits.split(data[:3000]):
        train_label = data[training_set, 0]
        train_samp = data[training_set, 1:]
        #Fit only to the training data
        scaler.fit(train_samp)
        train_samp = scaler.transform(train_samp)
        val_label = data[validation_set, 0]
        val_samp = data[validation_set, 1:]
        val_samp = scaler.transform(val_samp)
        classifier.fit(train_samp, train_label)
        val_pred = classifier.predict(val_samp)
        cm = confusion_matrix(val_pred, val_label)
        acc_list.append(accuracy(cm))
        print("Accuracy of MLPClassifier: ", accuracy(cm))
    return classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html


def optimizer(classifier, data):
    distr = {#'learning_rate_init': [0.1, 0.01, 0.001, 0.0001, 0.00001],
             'max_iter': [20],
             'batch_size': [1, 10, 50, 100, 500, 1000, 2000]
             #'hidden_layer_sizes': [(10,), (40,), (70,), (100,), (100, 50), (50, 10)]
            }
    clf = RandomizedSearchCV(classifier, distr, random_state=0, cv=4, scoring='neg_mean_squared_error')
    X = data[:8000, 1:]
    y = data[:8000, 0]
    labels = np.unique(y)
    lb = LabelBinarizer()
    lb.fit(labels)
    y = lb.transform(y)
    clf.fit(X, y)
    df = pd.DataFrame(clf.cv_results_)
    params_est = clf.best_params_
    print(params_est)


def testing(trained_model):
    with open("mnist_test.csv", "r") as file:
        reader = csv.reader(file)
        data = list(reader)  # converts csv.reader object into list of lists, each list being one line from the file
        data = np.array(data[1:], dtype=int)  # converts list of lists into an array of type int
        X = data[:, 1:]
        y = data[:, 0]
        X_pred = classifier.predict(X)
        cm = confusion_matrix(X_pred, y)
        print("Accuracy of MLPClassifier: ", accuracy(cm))





dat = load_data()
classif = classifier()

#loss_curve(dat, classif)
#new_params = optimizer(classif, dat)
# {'max_iter': 200, 'learning_rate_init': 0.001, 'hidden_layer_sizes': (100,)} for standard batch size
# {'max_iter': 20, 'batch_size': 50}
curve(dat, classif)
model = training(dat, classif)
testing(model)
