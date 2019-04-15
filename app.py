import sys
import os
import logging
import coloredlogs
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm #Support Vector Classification - estimador (implementa fit() y predict())


logger = logging.getLogger(__name__)


def initLogger(level='INFO'):
    coloredlogs.install(fmt='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S', level=level)
    logger.debug('Logger is active')


def main():
    iris = load_iris()

    # print (iris.data)
    # print (iris.feature_names)
    # print (iris.target)
    # print (iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

    #clasificador (utilizar frid search & cross validation para encontrar gamma)
    clf = svm.SVC(gamma=0.001, C=100.)

    model = clf.fit (X_train, y_train)

    logger.debug ("Predicción: {output}".format(output = clf.predict(X_test)))

    logger.debug ("Resultado real: {output}".format(output = y_test))

    logger.debug ("Score: {output}".format(output = model.score(X_test, y_test)))

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
