import sys
import os
import logging
import coloredlogs
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score


logger = logging.getLogger(__name__)


def initLogger(level='INFO'):
    coloredlogs.install(fmt='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S', level=level)
    logger.debug('Logger is active')

def classifierResults(clf, dataset):

    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target)

    model = clf.fit(X_train, y_train)

    predict = clf.predict(X_test)

    logger.debug("Predicción:\n{output}".format(output = predict))

    logger.debug("Resultado real:\n{output}".format(output = y_test))

    logger.debug("Score: {output}".format(output = model.score(X_test, y_test)))

    logger.debug("Matriz de confusión:\n{output}".format(output = confusion_matrix(y_test, predict, labels=[0, 1 ,2])))

    logger.debug("Accuracy: {output}".format(output = accuracy_score(y_test, predict)))

    logger.debug("Recall: {output}".format(output = recall_score(y_test, predict, average=None)))

    logger.debug("Precision: {output}".format(output = precision_score(y_test, predict, average=None)))

    logger.debug("Cross validation score: {output}".format(output = cross_val_score(clf, dataset.data, dataset.target, cv=5)))

def main():

    iris = load_iris()

    # print (iris.data)
    # print (iris.feature_names)
    # print (iris.target)
    # print (iris.target_names)

    #Support Vector Classification
    #Utilizar frid search & cross validation para encontrar gamma adecuado

    logger.debug("------Algorithm: Decision Tree------")

    clf = tree.DecisionTreeClassifier()
    classifierResults(clf, iris)

    logger.debug("------Algorithm: Support Vector Classification------")

    clf = svm.SVC(gamma=0.001, C=100.)
    classifierResults(clf, iris)

    logger.debug("------Algorithm: Gaussian Naive Bayes------")

    clf = GaussianNB()
    classifierResults(clf, iris)

    logger.debug("------Algorithm: Neuronal Netowrk------")

    clf = MLPClassifier(max_iter=1000)
    classifierResults(clf, iris)


if __name__ == '__main__':
    initLogger('DEBUG')

    try:
        main()
    except KeyboardInterrupt:
        logger.warning('Keyboard Interrupt... Exiting')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
