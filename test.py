##################################
# Exercise 2b: MLP               #
# Authors: Daniel Hanhart        #
#          Mauro Gwerder         #
#          Anika John            #
#          Liliane Trafelet      #
#                                #
##################################

# SOURCES FOR SKLEARN:
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
# https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2
# https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
# https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py

# SOURCES FOR KERAS (not utilized):
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
# https://analyticsindiamag.com/a-beginners-guide-to-scikit-learns-mlpclassifier/
# https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3

# GENERAL UNDERSTANDING:
# http://www.deeplearningbook.org/


import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, log_loss
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
import csv
from os import listdir


#https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
# Simple function for reading in .csv-files
def load_data(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        data = list(reader)  # converts csv.reader object into list of lists, each list being one line from the file
        matrix = np.array(data[1:], dtype=int)  # converts list of lists into an array of type int
        return matrix

# Utilization of the MLPclassifier-function form sklearn to create the classifier
# Parameters chosen by optimization: hidden_layer_sizes, max_iter, learning_rate_init
def classifier(max_iterations=400):
    classifier = MLPClassifier(hidden_layer_sizes=(100, ),
                               alpha=0.0001,
                               verbose=True,
                               activation='logistic',
                               solver='sgd',
                               max_iter=max_iterations,
                               random_state=1,
                               learning_rate='constant',
                               learning_rate_init=0.001,
                               batch_size=50,
                               )
    return classifier


# Using learning_curve() from sklearn to plot the learning function and the loss function
# by learning with five different training sizes (given by argument train_sizes) and
# cross-validation with four splits (given by arg cv=4)
def curve(data, classifier, title):
    X = data[:, 1:]  # removing labels from arrays
    y = data[:, 0]  # takes only labels
    labels = np.unique(y)  # all possible labels (0-9)
    lb = LabelBinarizer()
    lb.fit(labels)
    y = lb.transform(y)  # binarizes label such that they become categorical

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    # https://scikit-learn.org/stable/modules/learning_curve.html
    train_sizes, train_scores, valid_scores = learning_curve(classifier,
                                                                           X, y,
                                                                           train_sizes=np.linspace(.1, 1.0, 5),
                                                                           cv=4)
    train_mean = np.mean(train_scores, axis=1)  # mean of all cross-validation runs for training
    train_loss = 1 - train_mean  # loss = 1 - accuracy

    valid_mean = np.mean(valid_scores, axis=1)   # mean of all cross-validation runs for validation
    valid_loss = 1 - valid_mean  # loss = 1 - accuracy

    # plotting:
    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, train_mean, label="Training set")
    plt.plot(train_sizes, valid_mean, label="Validation set")
    plt.legend()
    plt.title("Learning Curve: Accuracy")
    plt.ylabel("Score")
    plt.xlabel("Images")
    plt.subplot(1,2,2)
    plt.plot(train_sizes, train_loss, label="Training set")
    plt.plot(train_sizes, valid_loss, label="Validation set")
    plt.legend()
    plt.title("Learning Curve: Loss")
    plt.ylabel("Score")
    plt.xlabel("Images")
    plt.suptitle(title, fontsize=14)
    plt.show()


def epoch_curve(data, classif, title):
    y = data[:45000, 0]

    # Make labels categorical using One-Hot-Encoding
    labels = np.unique(y)  # all possible labels (0-9)
    lb = LabelBinarizer()
    lb.fit(labels)
    y = lb.transform(y)

    y_val = data[45000:, 0]
    y_val = lb.transform(y_val)
    X = data[:45000, 1:]
    X_val = data[45000:, 1:]
    N_TRAIN_SAMPLES = X.shape[0]
    N_EPOCHS = 400
    N_BATCH = 50
    N_CLASSES = np.unique(y, axis=1)

    scores_train = []
    loss_train = []
    scores_val = []
    loss_val = []
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        # Shuffling
        random_perm = np.random.permutation(X.shape[0])
        mini_batch_index = 0
        while True:
            # Mini batches and partial fitting
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            classif.partial_fit(X[indices], y[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # Score train
        scores_train.append(classif.score(X, y))
        y_pred_train = classif.predict(X)
        loss_train.append(log_loss(y, y_pred_train))  # calculate cross-entropy for loss-curve
        # Score test
        scores_val.append(classif.score(X_val, y_val))
        y_pred_val = classif.predict(X_val)
        loss_val.append(log_loss(y_val, y_pred_val))  # calculate cross-entropy for loss-curve
        epoch += 1

    # Plotting
    plt.subplot(1,2,1)
    plt.plot(scores_train, label='Training Set')
    plt.plot(scores_val, label='Validation Set')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(loss_train, label='Training Set')
    plt.plot(loss_val, label='Validation Set')
    plt.title('Loss Curve')
    plt.legend()
    plt.suptitle(title)
    plt.show()



# function needed to calculate the accuracy from the output of the function confusion_matrix()
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


# Training with cross-validation
def training(data, classifier):
    scaler = StandardScaler()
    splits = KFold(4)  # amounts of splits in the dataset for cross-validation

    for training_set, validation_set in splits.split(data):
        y = data[training_set, 0]
        labels = np.unique(y)  # all possible labels (0-9)
        X = data[training_set, 1:]
        #Fit only to the training data
        scaler.fit(X)
        X = scaler.transform(X)
        y_val = data[validation_set, 0]
        X_val = data[validation_set, 1:]
        X_val = scaler.transform(X_val)
        classifier.fit(X, y)
        y_pred = classifier.predict(X_val)  # predict with created model
        cm = confusion_matrix(y_pred, y_val)
        print("Accuracy of MLPClassifier: ", accuracy(cm))
    return classifier


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html
def optimizer(classifier, data):
    # Values for the parameters which are going to be tested
    distr = {'learning_rate_init': [0.1, 0.01, 0.001, 0.0001, 0.00001],
             'max_iter': [300, 400, 500, 600],
             'hidden_layer_sizes': [(10,), (40,), (70,), (100,), (100, 50), (50, 10)]
             }
    # Doing a randomized search with cross-validation = 4
    clf = RandomizedSearchCV(classifier, distr, random_state=0, cv=4, scoring='neg_mean_squared_error')
    X = data[:, 1:]
    y = data[:, 0]
    labels = np.unique(y)  # see explanation in curve()
    lb = LabelBinarizer()
    lb.fit(labels)
    y = lb.transform(y)
    clf.fit(X, y)  # Apply randomized search on to training set
    params_est = clf.best_params_
    print(params_est)


# Prediction of the test-dataset with the trained model.
def testing(trained_model, data):
        X = data[:, 1:]
        y = data[:, 0]

        y_pred = trained_model.predict(X)
        cm = confusion_matrix(y_pred, y)
        print("Accuracy of MLPClassifier: ", accuracy(cm))


# Function to load data directly from folders, as given to us.
# Returns an array that has the same structure as when loading them from a .csv-file.
def permut_load():
    folder_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    datasets = ['train', 'test', 'val']
    train_csv = []
    test_csv = []
    val_csv = []
    lists = [train_csv, test_csv, val_csv]
    path = '../'  # Your own path to the folders 'train', 'test', 'val'
    for i in np.arange(len(datasets)):
        for folder in folder_names:
            mypath = path + datasets[i] + '/' + folder  # concatenates current path
            filenames = [f for f in listdir(mypath)]  # extracts all filenames from folder
            for file in filenames:
                img = np.asarray(plt.imread(mypath + '/' + file))
                img = img[..., 1].reshape(1, 784)[0]  # plt.imread() converts grayscale-img to RGB, so this is a way to get back to grayscale
                tagged_img = np.concatenate([np.asarray([int(folder)]), img])  # concatenates label in front of image
                lists[i].append(tagged_img)
    train_csv = np.random.permutation(np.asarray(train_csv))
    test_csv = np.random.permutation(np.asarray(test_csv))
    val_csv = np.random.permutation(np.asarray(val_csv))
    return train_csv, test_csv, val_csv


if __name__ == '__main__':
    dat = load_data("mnist_train.csv")
    classif = classifier()

    #loss_curve(dat, classif)
    #new_params = optimizer(classif, dat)
    # {'max_iter': 200, 'learning_rate_init': 0.001, 'hidden_layer_sizes': (100,)} for standard batch size
    # curve(dat, classif, title="Standard MNIST Dataset learning curves")
    model = training(dat, classif)
    # Accuracy of MLPClassifier:  0.963
    test_dat = load_data("mnist_test.csv")
    testing(model, test_dat)
    # Accuracy of MLPClassifier:  0.8971, 0.9629

    #p_train, p_test, p_val = permut_load()
    #curve(p_train, classif, title="Permutated MNIST Dataset learning curves")
    #p_train = np.concatenate([p_train, p_val])
    #print(p_train.shape)
    #model_permut = training(p_train, classif)
    #testing(model_permut, p_test)
    #epoch_curve(dat, classifier(max_iterations=1), title="Standard MNIST Dataset learning curves")
    #epoch_curve(p_train, classifier(max_iterations=1), title="Permutated MNIST Dataset learning curves")
