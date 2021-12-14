########################################################################################################################
#
# Author: David Schwartz, June, 9, 2020
#
# This file pretrains an instance of VGG for a specified dataset to be used in the fine tuning experiment. 
########################################################################################################################

########################################################################################################################
# imports
from HLDR_VGG_16 import HLDRVGG
from HLRGD_VGG_16 import HLRGDVGG
import time
import pickle
import os, sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# tf.executing_eagerly()
import itertools
import random
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

# %matplotlib inline
# import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from sklearn import model_selection
# from robust_attacks import l2_attack
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
# tf.compat.v1.disable_eager_execution()
########################################################################################################################


########################################################################################################################
# define parameters
outputModelPath = '../'
optimizer=None#uses nadam
verbose = True
testFraction = 0.25
numberOfClasses = 10#2
kerasBatchSize = 128
trainingSetSize =10000#-1#10000 for fashion mnist
dataset = 'fashion_mnist'#commented params are fashion mnist
numberOfAdvSamples = 100
trainingEpochs = 2
dropoutRate = 0.33
chosenPatience = trainingEpochs
powers = [0.01, 0.025, 0.05, 0.1, 0.25]
print("powers: %s"%str(powers))
########################################################################################################################

########################################################################################################################
# input image dimensions
img_rows, img_cols = 32, 32#28, 28

# split data between train and test sets
if (dataset == 'fashion_mnist'):
    input_shape = (img_rows, img_cols, 1)
    unprotectedModelName = ''.join(['unprotectedFashionMNIST', '_%s_' % str(time.time()), '.h5'])
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train_up, x_test_up = np.zeros((x_train.shape[0], 32, 32)), np.zeros((x_test.shape[0], 32, 32))
    # upscale data to 32,32 (same size as cifar)
    for i in range(x_train.shape[0]):
        x_train_up[i,:,:] = cv2.resize(x_train[i,:,:], (32, 32), interpolation = cv2.INTER_AREA)
    for i in range(x_test.shape[0]):
        x_test_up[i,:,:] = cv2.resize(x_test[i,:,:], (32, 32), interpolation = cv2.INTER_AREA)
    x_train = x_train_up
    x_test = x_test_up

    #scale and standardize (with respect to stats calculated on training set)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.


elif (dataset ==     'cifar10'):
    input_shape = (img_rows, img_cols, 3)
    unprotectedModelName = ''.join(['unprotectedCifar10VGG', '_%s_' % str(time.time()), '.h5'])
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # scale and standardize (with respect to stats calculated on training set)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.


# restrict number of classes
trainXs, trainTargets = [], []
testXs, testTargets = [], []

for t in range(numberOfClasses):
    curClassIndicesTraining = np.where(y_train == t)[0]
    curClassIndicesTesting = np.where(y_test == t)[0]
    if (trainingSetSize == -1):
        # arrange training data
        # curXTrain = np.squeeze(x_train[curClassIndicesTraining, :])
        if (dataset == 'fashion_mnist'):
            curXTrain = np.expand_dims(x_train[curClassIndicesTraining, :, :], axis=3)
            curXTest = np.expand_dims(x_test[curClassIndicesTesting, :, :], axis=3)
        else:
            curXTrain = x_train[curClassIndicesTraining, :, :]
            curXTest = x_test[curClassIndicesTesting, :, :]

    else:
        if (dataset == 'fashion_mnist'):
            # arrange training data
            curXTrain = np.expand_dims(x_train[curClassIndicesTraining[:trainingSetSize], :, :], axis=3)
            # arrange testing data
            curXTest = np.expand_dims(x_test[curClassIndicesTesting[:trainingSetSize], :, :], axis=3)
        else:
            # arrange training data
            curXTrain = x_train[curClassIndicesTraining[:trainingSetSize], :, :]
            # arrange testing data
            curXTest = x_test[curClassIndicesTesting[:trainingSetSize], :, :]

    trainXs.append(curXTrain)
    trainTargets.append((t * np.ones([curXTrain.shape[0], 1])))

    testXs.append(curXTest)
    testTargets.append(t * np.ones([curXTest.shape[0], 1]))
# stack our data
t = 0
stackedData, stackedTargets = np.array([]), np.array([])
# stackedData = np.concatenate((stackedData, trainXs[t]), axis=0) if stackedData.size > 0 else trainXs[t]
# stackedTargets = np.concatenate((stackedTargets, trainTargets[t]), axis=0) if stackedTargets.size > 0 else trainTargets[t]
# stackedData = np.concatenate((stackedData, testXs[t]), axis=0) if stackedData.size else testXs[t]
# stackedTargets = np.concatenate((stackedTargets, testTargets[t]), axis=0) if stackedTargets.size else testTargets[t]
for t in range(numberOfClasses):
    if (verbose):
        print('current class count')
        print(t + 1)
    stackedData = np.concatenate((stackedData, trainXs[t]), axis=0) if stackedData.size > 0 else trainXs[t]
    stackedData = np.concatenate((stackedData, testXs[t]), axis=0)
    stackedTargets = np.concatenate((stackedTargets, trainTargets[t]), axis=0) if stackedTargets.size > 0 else \
    trainTargets[t]
    stackedTargets = np.concatenate((stackedTargets, testTargets[t]), axis=0)

trainX, testX, trainY, testY = model_selection.train_test_split(stackedData, stackedTargets, shuffle=True,
                                                                test_size=testFraction, random_state=42)
trainX, valX, trainY, valY = model_selection.train_test_split(trainX, trainY, shuffle=True,
                                                                test_size=0.05, random_state=43)

defaultLossThreshold = 0.001

unprotectedModel = HLDRVGG(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0.0005,
                       loss_threshold=defaultLossThreshold, patience=chosenPatience, dropout_rate=dropoutRate,
                       max_relu_bound=1.1, optimizer=optimizer, verbose=False, unprotected=True)
protectedModel = HLDRVGG(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0.0005,
                       loss_threshold=defaultLossThreshold, patience=chosenPatience, dropout_rate=dropoutRate,
                       max_relu_bound=1.1, optimizer=optimizer, verbose=False)
loss, acc, bestModelPath = unprotectedModel.train([trainX, trainX], tf.keras.utils.to_categorical(trainY, numberOfClasses),
                                                  [valX, valX], tf.keras.utils.to_categorical(valY, numberOfClasses),
                                                  training_epochs=trainingEpochs, monitor='val_loss', patience=1000,
                                                  model_path=outputModelPath, keras_batch_size=kerasBatchSize, dataAugmentation=True)
#unprotectedModel.storeModelToDisk(unprotectedModelName)
unprotectedModel.model.save_weights(unprotectedModelName)
unprotectedModel.model.load_weights(unprotectedModelName)
#unprotectedModel.model.save(unprotectedModelName)

print("unprotected model saved to disk")

#examine performances on benign data
benignUnProtEval = unprotectedModel.evaluate([testX, testX], tf.keras.utils.to_categorical(testY))
benignUnProtAcc = benignUnProtEval[1]
print('benign unprotected acc: %s'%str(benignUnProtAcc))
########################################################################################################################
