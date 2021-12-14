########################################################################################################################
#
# Author: David Schwartz, June, 9, 2020
#
# This file implements the fine tuning cross validation experiment discussed in the paper.
########################################################################################################################

########################################################################################################################
# imports
from HLDR_VGG_16 import HLDRVGG
from HLRGD_VGG_16 import HLRGDVGG
import os, sys, time, copy
import argparse
from itertools import chain
import pickle
import cv2
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from sklearn import model_selection
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import _safe_indexing as safe_indexing, indexable
########################################################################################################################

########################################################################################################################
#some training parameters
chosenOptimizer = None#defaults to Nadam
testingStability = False
defaultPatience = 50
defaultLossThreshold = 0.001
img_rows, img_cols = 32, 32#28, 28
#set random seeds once and minimize stochasticity from gpu processing 
seed = 42
tf.keras.backend.clear_session()
os.environ['PYTHONHASHSEED']=str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(a=seed, version=2)
tf.random.set_seed(seed)

#select the dataset
dataset = 'cifar10'#cifar10'#'fashion_mnist'

# define parameters as a function of the chosen dataset
if (dataset == 'fashion_mnist'):
    HLDRReg = reg='HLDR'
    unprotectedModelName = 'unprotectedFashionMNIST.h5'
    verbose = True
    noisyTraining = True
    noisyTrainingSigma = 32./255
    numberOfAGNSamples = 1
    numberOfReplications = 10
    testFraction = 0.25
    numberOfClasses = 10
    kerasBatchSize = 64
    validationSplit =0.25
    penaltyCoefficientHLDR = 0.25
    penaltyCoefficientFIM = 0.00025
    trainingSetSize = 10000
    numberOfAdvSamplesTrain = 1024
    numberOfAdvSamplesTest = 100
    trainingEpochs = 10
    dropoutRate = 0
    powersAcc = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
    noisePowers = list(np.arange(0.05, 0.35, 0.1))
    trainingBudgetMax = 32./255
    adversarialMAEBudget = 8./255
    trainingBudgets = [0, trainingBudgetMax/64, trainingBudgetMax/32, trainingBudgetMax/16, trainingBudgetMax]
    numberOfTrainingBudgets = len(trainingBudgets)

elif(dataset == 'cifar10'):
    HLDRReg = reg = 'HLDR'
    unprotectedModelName = 'deeperVGGCifar.h5'#'unprotectedCifar10VGG.h5'
    verbose = True
    noisyTraining = True
    noisyTrainingSigma = 32./255
    numberOfAGNSamples = 1
    numberOfReplications = 5
    testFraction = 0.25
    numberOfClasses = 10
    kerasBatchSize = 512
    validationSplit = 0.25
    penaltyCoefficientHLDR = 0.5
    penaltyCoefficientFIM = 0.00025
    trainingSetSize = -1
    numberOfAdvSamplesTrain = 2048
    numberOfAdvSamplesTest = 128
    trainingEpochs = 10
    dropoutRate = 0
    powersAcc = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
    noisePowers = list(np.arange(0.05, 0.35, 0.1))
    trainingBudgetMax = 32./255
    adversarialMAEBudget = 8./255
    trainingBudgets = [0, trainingBudgetMax/16, trainingBudgetMax]
    # trainingBudgets = [0, trainingBudgetMax/64, trainingBudgetMax/4]
    numberOfTrainingBudgets = len(trainingBudgets)

########################################################################################################################

########################################################################################################################
# split data between train and test sets
if (dataset == 'fashion_mnist'):
    input_shape = (img_rows, img_cols, 1)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train_up, x_test_up = np.zeros((x_train.shape[0], 32, 32)), np.zeros((x_test.shape[0], 32, 32))
    # upscale data to 32,32 (same size as cifar)
    for i in range(x_train.shape[0]):
        x_train_up[i,:,:] = cv2.resize(x_train[i,:,:], (32, 32), interpolation = cv2.INTER_AREA)
    for i in range(x_test.shape[0]):
        x_test_up[i,:,:] = cv2.resize(x_test[i,:,:], (32, 32), interpolation = cv2.INTER_AREA)
    x_train = x_train_up
    x_test = x_test_up

    # scale the data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.


elif (dataset == 'cifar10'):
    input_shape = (img_rows, img_cols, 3)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # scale the data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

#normalize the data once after the pretraining experiment
mean = np.mean(x_train)
std = np.std(x_train)
print(mean)
print(std)
x_train -= mean
x_test -= mean
x_train /= std
x_test /= std
print(x_test.shape)
print(x_train.shape)


# restrict number of classes
trainXs, trainTargets = [], []
testXs, testTargets = [], []

for t in range(numberOfClasses):
    curClassIndicesTraining = np.where(y_train == t)[0]
    curClassIndicesTesting = np.where(y_test == t)[0]
    if (trainingSetSize == -1):
        # arrange training data
        if (dataset == 'fashion_mnist'):
            curXTrain = np.expand_dims(x_train[curClassIndicesTraining, :, :], axis=3)
            curXTest = np.expand_dims(x_test[curClassIndicesTesting, :, :], axis=3)
        else:
            curXTrain = x_train[curClassIndicesTraining, :, :]
            curXTest = x_test[curClassIndicesTesting, :, :]
        # arrange testing data
        # curXTest = np.squeeze(x_test[curClassIndicesTesting, :])

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
for t in range(numberOfClasses):
    if (verbose):
        print('current class count')
        print(t + 1)
    stackedData = np.concatenate((stackedData, trainXs[t]), axis=0) if stackedData.size > 0 else trainXs[t]
    stackedData = np.concatenate((stackedData, testXs[t]), axis=0)
    stackedTargets = np.concatenate((stackedTargets, trainTargets[t]), axis=0) if stackedTargets.size > 0 else \
    trainTargets[t]
    stackedTargets = np.concatenate((stackedTargets, testTargets[t]), axis=0)

#since this experiment resumes after pre-training, we only want our test data, since in part 1, we trained on our training data
trainX, testX, trainY, testY = model_selection.train_test_split(stackedData, stackedTargets, shuffle=True,
                                                                test_size=testFraction, random_state=42)

trainX, testX, trainY, testY = copy.copy(trainX), copy.copy(testX), copy.copy(trainY), copy.copy(testY)
stackedData = testX
stackedTargets = testY

#if we are only running one replication of the CV experiment
if (numberOfReplications == 1):
    splitIndices = [(range(stackedData.shape[0])[:np.floor(stackedData.shape[0]/2.).astype(int)], range(stackedData.shape[0])[(np.floor(stackedData.shape[0]/2.).astype(int)+1):])]
#if k > 1
else:
    kFold = StratifiedKFold(n_splits=numberOfReplications, shuffle=True, random_state=42)
    splitIndices = [(trainIndices, testIndices) for trainIndices, testIndices in kFold.split(stackedData, stackedTargets)]
########################################################################################################################

########################################################################################################################
#execute the cross val experiment
crossValResults = dict()
crossValResults['Acc'] = dict()
crossValResults['noiseAcc'] = dict()
crossValResults['splitIndices'] = dict()
crossValResults['averageLayerwiseUndefMAEs'] = dict()
crossValResults['averageLayerwiseHLDRMAEs'] = dict()
crossValResults['averageLayerwiseHLRGDMAEs'] = dict()
crossValResults['averageLayerwiseFIMMAEs'] = dict()
crossValResults['averageLayerwiseAGNMAEs'] = dict()
crossValResults['averageLayerwiseAGNUndefMAEs'] = dict()
crossValResults['averageLayerwiseAGNHLDRMAEs'] = dict()
crossValResults['averageLayerwiseAGNHLRGDMAEs'] = dict()
crossValResults['averageLayerwiseAGNFIMMAEs'] = dict()
crossValResults['averageLayerwiseAGNAGNMAEs'] = dict()
crossValResults['budgets'] = powersAcc
crossValResults['noiseBudgets'] = noisePowers
crossValResults['noisyTraining'] = noisyTraining
cvIndex = 0

for trainIndices, testIndices in splitIndices:

    #slice this iteration's data
    trainX, testX = stackedData[trainIndices, :, :, :], stackedData[testIndices, :, :, :]
    trainY, testY = stackedTargets[trainIndices, :], stackedTargets[testIndices, :]

    print("%s total training indices"%str(len(trainIndices)))
    crossValResults['splitIndices'][cvIndex] = (trainIndices, testIndices)


    unprotectedModel = HLDRVGG(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0, optimizer=chosenOptimizer,
                               loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                               max_relu_bound=1.1, verbose=False, unprotected=True)
    unprotectedModel.model.load_weights(unprotectedModelName)
    print("unprotected model loaded from disk")

    # define fgsm method
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    def create_adversarial_pattern(input_image, input_label):
        inputImageTensor = tf.cast(input_image, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(inputImageTensor)
            prediction = unprotectedModel.model([inputImageTensor, inputImageTensor])
            loss = loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, inputImageTensor)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad
    

    #set up containers to store perturbations
    fgsmData = dict()
    awgnData = dict()
    fgsmData['train'] = np.zeros((numberOfAdvSamplesTrain, 32, 32, 1))
    fgsmData['test'] = np.zeros((numberOfAdvSamplesTest, 32, 32, 1))
    awgnData['test'] = dict()
    # generate awgn noise
    for i in range(len(noisePowers)):
        awgnData['test'][i] = np.random.normal(0, scale=noisePowers[i], size=testX.shape)
    print("awg noise generated")

    #attack the unprotected model
    fgsmData['train'] = K.eval(create_adversarial_pattern(trainX[:numberOfAdvSamplesTrain, :, : ,:], tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], num_classes=numberOfClasses)))
    fgsmData['test'] = K.eval(create_adversarial_pattern(testX[:numberOfAdvSamplesTest, :, : ,:], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses)))
    print("attack complete")

    #form the combined multibudget training set
    trainingAttacks = np.array([])
    trainingBenigns = np.array([])
    distTrainingLabels = np.array([])
    for curTrainingBudget in trainingBudgets:
        curTrainingAttacks = trainX[:numberOfAdvSamplesTrain, :, :, :] + curTrainingBudget*fgsmData['train']
        trainingAttacks = np.concatenate((trainingAttacks, curTrainingAttacks), axis=0) if trainingAttacks.size > 0 else curTrainingAttacks
        trainingBenigns = np.concatenate((trainingBenigns, trainX[:numberOfAdvSamplesTrain, :, :, :]), axis=0) if trainingBenigns.size > 0 else trainX[:numberOfAdvSamplesTrain, :, :, :]
        distTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), distTrainingLabels), axis=0) if distTrainingLabels.size > 0 \
            else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)

    #advOnlyAttacks, advOnlyBenigns, advOnlyDistLabels = trainingAttacks, trainingBenigns, distTrainingLabels

    
	
    #this block includes gaussianly noisy data to be included in the attack component of this training set
    if (noisyTraining):
        noiseOnlyTrainingAttacks, noiseOnlyTrainingBenigns, noiseOnlyTrainingLabels = np.array([]), np.array([]), np.array([])
        for curAGNSample in range(numberOfAGNSamples):
            curNoisyTrainingSamples = trainX[:numberOfAdvSamplesTrain, :, :, :] + np.random.normal(0, scale=noisyTrainingSigma, size=(numberOfAdvSamplesTrain, trainX.shape[1], trainX.shape[2], trainX.shape[3]))

            noiseOnlyTrainingAttacks = np.concatenate((noiseOnlyTrainingAttacks, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingAttacks.size > 0 else curNoisyTrainingSamples
            noiseOnlyTrainingBenigns = np.concatenate((noiseOnlyTrainingBenigns, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingBenigns.size > 0 else curNoisyTrainingSamples
            noiseOnlyTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), noiseOnlyTrainingLabels), axis=0) if noiseOnlyTrainingLabels.size > 0 \
                   else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)
            noiseOnlyTrainingAttacks = np.concatenate((noiseOnlyTrainingAttacks, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingAttacks.size > 0 else curNoisyTrainingSamples
            noiseOnlyTrainingBenigns = np.concatenate((noiseOnlyTrainingBenigns, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingBenigns.size > 0 else curNoisyTrainingSamples
            noiseOnlyTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), noiseOnlyTrainingLabels), axis=0) if noiseOnlyTrainingLabels.size > 0 \
                   else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)

            #form exclusive awgn training data
            for z in range(numberOfTrainingBudgets):
                noiseOnlyTrainingAttacks = np.concatenate((noiseOnlyTrainingAttacks, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingAttacks.size > 0 else curNoisyTrainingSamples
                noiseOnlyTrainingBenigns = np.concatenate((noiseOnlyTrainingBenigns, trainX[:numberOfAdvSamplesTrain, :, :, :]), axis=0) if noiseOnlyTrainingBenigns.size > 0 else curNoisyTrainingSamples
                noiseOnlyTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), noiseOnlyTrainingLabels), axis=0) if noiseOnlyTrainingLabels.size > 0 \
                    else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)

            #append disparity regularized noisy training data to other models adv training sets#
            trainingAttacks = np.concatenate((trainingAttacks, curNoisyTrainingSamples), axis=0) if trainingAttacks.size > 0 else curNoisyTrainingSamples
            trainingBenigns = np.concatenate((trainingBenigns, trainX[:numberOfAdvSamplesTrain, :, :, :]), axis=0) if trainingBenigns.size > 0 else None
            distTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), distTrainingLabels), axis=0) if distTrainingLabels.size > 0 \
               else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)
            #append disparity un-regularized noisy training data to other models adv training sets#
            trainingAttacks = np.concatenate((trainingAttacks, curNoisyTrainingSamples), axis=0) if trainingAttacks.size > 0 else curNoisyTrainingSamples
            trainingBenigns = np.concatenate((trainingBenigns, curNoisyTrainingSamples), axis=0) if trainingBenigns.size > 0 else None
            distTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), distTrainingLabels), axis=0) if distTrainingLabels.size > 0 \
               else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)

    #print fine tuning dataset dimensionality and size
    if (noisyTraining):
        print("%s gaussianly noisy training samples generated" % str(noiseOnlyTrainingAttacks.shape[0]))
    print("%s adv. training samples generated"%str(trainingAttacks.shape[0]))
    testingAttacks = testX[:numberOfAdvSamplesTest, :, :, :] + np.max(trainingBudgets) * fgsmData['test']
    print("testing attacks generated")

    #split the data into training and validation subsets
    preSplitData, preSplitTargets = [trainingBenigns, trainingAttacks], distTrainingLabels
    preSplitNoiseData = [noiseOnlyTrainingAttacks, noiseOnlyTrainingAttacks]
    #preSplitAdvOnlyData, preSplitAdvOnlyTargets = [advOnlyAttacks, advOnlyBenigns], advOnlyDistLabels
    if (validationSplit > 0):
        #split the data (here, so we don't risk having slightly different training sets for different models) for validation
        sss = StratifiedShuffleSplit(n_splits=1, test_size=validationSplit, random_state=0)
        arrays = indexable(preSplitData[0], preSplitTargets)
        train, test = next(sss.split(X=preSplitData[0], y=preSplitTargets))
        iterator = list(chain.from_iterable((safe_indexing(a, train),
                                                safe_indexing(a, test),
                                                train,
                                                test) for a in arrays))
        X_train, X_val, train_is, val_is, y_train, y_val, _, _ = iterator
        trainingData, validationData = [], []
        trainingData = [np.copy(preSplitData[0][train_is,:,:]), np.copy(preSplitData[1][train_is, :, :])]
        validationData = [np.copy(preSplitData[0][val_is,:,:]), np.copy(preSplitData[1][val_is, :, :])]
        awgnTrainingData = [np.copy(preSplitNoiseData[0][train_is,:,:]), np.copy(preSplitNoiseData[1][train_is, :, :])]
        awgnValidationData = [np.copy(preSplitNoiseData[0][val_is,:,:]), np.copy(preSplitNoiseData[1][val_is, :, :])]

        trainingTargets, validationTargets = np.copy(y_train), np.copy(y_val)

    else:
        print("set validation split fraction (validationSplit)")
    
    print("Data are split")

    #we're not training the undefended model, but we need to initialize its scaler
    unprotectedModel.normalize(trainingData[0], storeStats=True)

    #instantiate our models
    protectedModel = HLDRVGG(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=penaltyCoefficientHLDR,
                             loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                             max_relu_bound=1.1, reg=HLDRReg, verbose=False)
    eigenRegModel = HLDRVGG(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=penaltyCoefficientFIM,
                             loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                             max_relu_bound=1.1, reg='FIM', verbose=False)
    hlrgd = HLRGDVGG(input_shape, numberOfClasses, vggModelLocation=os.path.join(os.getcwd(),unprotectedModelName),
                        number_of_classes=numberOfClasses, adv_penalty=0.0005, 
                        loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=0,
                        max_relu_bound=1.1, verbose=False)                             
    awgnModel = HLDRVGG(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0,
                               loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                               max_relu_bound=1.1, verbose=False, unprotected=True)

    #train an HLDR protected model (ours)
    protectedModel.model.set_weights(copy.copy(unprotectedModel.model.get_weights()))
    loss, acc, bestProtectedModelPath = protectedModel.train(trainingData, trainingTargets, validationData, validationTargets, training_epochs=trainingEpochs,
                                                             monitor='val_loss', patience=defaultPatience, model_path=None, keras_batch_size=kerasBatchSize,
                                                             dataAugmentation=False, adversarialOrder=0, normed=True)
    protectedModel.model.load_weights(bestProtectedModelPath)

    #train an FIM regularization protected model (Shen's)
    #safe to give benign training set to this because this model ignores the second component in the loss calculation
    eigenRegModel.model.set_weights(copy.copy(unprotectedModel.model.get_weights()))
    loss, acc, bestEigenRegModelPath = eigenRegModel.train(trainingData, trainingTargets, validationData, validationTargets,
                                                           training_epochs=trainingEpochs, monitor='val_loss', patience=defaultPatience,\
                                                           model_path=None, keras_batch_size=kerasBatchSize,\
                                                           dataAugmentation=False, normed=True, adversarialOrder=0)
    if (not (np.isnan(loss).any() or np.isinf(loss).any())):
        eigenRegModel.model.load_weights(bestEigenRegModelPath)

    #train the AWGN model
    if (noisyTraining):
        #train the awgn only model
        awgnModel.model.set_weights(copy.copy(unprotectedModel.model.get_weights()))
        loss, acc, bestawgnModelPath = awgnModel.train(awgnTrainingData, trainingTargets, awgnValidationData, validationTargets, normed=True,
                                                        training_epochs=trainingEpochs, monitor='val_loss', patience=defaultPatience, 
                                                        model_path=None, keras_batch_size=kerasBatchSize, adversarialOrder=1, dataAugmentation=False)
        awgnModel.model.load_weights(bestawgnModelPath)

    #train an hlrgd
    hlrgd.train(trainingData, trainingTargets, validationData, validationTargets,
                training_epochs=trainingEpochs, normed=True, monitor='val_loss', adversarialOrder=0,
                patience=defaultPatience, model_path=None, keras_batch_size=kerasBatchSize, dataAugmentation=False)


    print("unprotected model")
    #evaluate the unprotected model
    attackResults = unprotectedModel.evaluate([testingAttacks, testingAttacks], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses), batchSize=kerasBatchSize)
    print(attackResults)
    benignResultsUnprotected = unprotectedModel.evaluate([testX, testX], tf.keras.utils.to_categorical(testY), batchSize=kerasBatchSize)
    print(benignResultsUnprotected)

    print("protected eigenreg model")
    #evaluate the protected model
    attackResults = eigenRegModel.evaluate([testingAttacks, testingAttacks], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses), batchSize=kerasBatchSize)
    print(attackResults) 
    benignResultsFIM = eigenRegModel.evaluate([testX, testX], tf.keras.utils.to_categorical(testY), batchSize=kerasBatchSize)
    print(benignResultsFIM)


    print("protected hldr")
    #evaluate the protected model
    attackResults = protectedModel.evaluate([testingAttacks, testingAttacks], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses), batchSize=kerasBatchSize)
    print(attackResults)
    benignResultsHLDR = protectedModel.evaluate([testX, testX], tf.keras.utils.to_categorical(testY), batchSize=kerasBatchSize)
    print(benignResultsHLDR)

    benignResultsAGN = [-1,-1]
    if (noisyTraining):
        print("awgn")
        #evaluate the awgn model
        attackResultsAGN = awgnModel.evaluate([testingAttacks, testingAttacks], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses), batchSize=kerasBatchSize)
        print(attackResultsAGN)
        benignResultsAGN = awgnModel.evaluate([testX, testX], tf.keras.utils.to_categorical(testY), batchSize=kerasBatchSize)
        print(benignResultsAGN)

    print("hlrgd")
    attackResults = hlrgd.evaluate([testingAttacks, testingAttacks], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses), batchSize=kerasBatchSize)
    print(attackResults)
    benignResultsHLRGD = hlrgd.evaluate([testX, testX], tf.keras.utils.to_categorical(testY), batchSize=kerasBatchSize)
    print(benignResultsHLRGD)

    #iterate over adversarial powers for HLDR, undefended, and hlrgd, and calculate adversarial accuracy for each of these
    advAccuracy, noiseAccuracy = dict(), dict()
    advAccuracy['undefended'], advAccuracy['hldr'], advAccuracy['fim'], advAccuracy['hlrgd'], advAccuracy['agn'] = \
        [benignResultsUnprotected[1]], [benignResultsHLDR[1]], [benignResultsFIM[1]], [benignResultsHLRGD[1]], [benignResultsAGN[1]]
    noiseAccuracy['undefended'], noiseAccuracy['hldr'], noiseAccuracy['fim'], noiseAccuracy['hlrgd'], noiseAccuracy['agn'] = \
        [benignResultsUnprotected[1]], [benignResultsHLDR[1]], [benignResultsFIM[1]], [benignResultsHLRGD[1]], [benignResultsAGN[1]]

    #set up containers to store perturbations
    # fgsmData['test'] = np.zeros((numberOfAdvSamplesTest, input_shape[0], input_shape[1], input_shape[2]))
    averageLayerwiseHLRGDMAEs, averageLayerwiseHLDRMAEs, averageLayerwiseUndefMAEs, averageLayerwiseFIMMAEs = dict(), dict(), dict(), dict()
    averageLayerwiseAGNHLRGDMAEs, averageLayerwiseAGNHLDRMAEs, averageLayerwiseAGNUndefMAEs, averageLayerwiseAGNFIMMAEs = dict(), dict(), dict(), dict()

    #attack the unprotected model
    # fgsmData['train'] = K.eval(create_adversarial_pattern(trainX[:numberOfAdvSamples, :, : ,:], tf.keras.utils.to_categorical(trainY[:numberOfAdvSamples], num_classes=numberOfClasses)))
    # fgsmData['test'] = K.eval(create_adversarial_pattern(testX[:numberOfAdvSamplesTest, :, : ,:], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses)))
    # print("attack data generated (%s samples)"%str(fgsmData['test'].shape[0]))

    #adversarial noise test
    evalLabels = tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses)#labels for the next few evaluation blocks, so we only need to call this once
    for curNoisePower in powersAcc:
        #calculate new testingAttacks
        testingAttacks = testX[:numberOfAdvSamplesTest, :, :, :] + curNoisePower*fgsmData['test']

        #get performances
        unprotectedResult = unprotectedModel.evaluate([testingAttacks, testingAttacks], evalLabels, batchSize=kerasBatchSize)[1]
        HLDRResult = protectedModel.evaluate([testingAttacks, testingAttacks], evalLabels, batchSize=kerasBatchSize)[1]
        hlrgdResult = hlrgd.evaluate([testingAttacks, testingAttacks], evalLabels, batchSize=kerasBatchSize)[1]
        fimResult = eigenRegModel.evaluate([testingAttacks, testingAttacks], evalLabels, batchSize=kerasBatchSize)[1]

        #collate reattackResults
        advAccuracy['undefended'].append(unprotectedResult)
        advAccuracy['hldr'].append(HLDRResult)
        advAccuracy['hlrgd'].append(hlrgdResult)
        advAccuracy['fim'].append(fimResult)

        #evaluate the awgn model
        if (noisyTraining):
            awgnResult = awgnModel.evaluate([testingAttacks, testingAttacks], evalLabels, batchSize=kerasBatchSize)[1]
            advAccuracy['agn'].append(awgnResult)


    #awg noise test
    for curNoiseIndex in range(len(noisePowers)):
        #calculate new testingAttacks
        corruptedTestX = testX[:numberOfAdvSamplesTest, :, :, :] + awgnData['test'][curNoiseIndex][:numberOfAdvSamplesTest, :, :, :]

        #get performances
        unprotectedResult = unprotectedModel.evaluate([corruptedTestX, corruptedTestX], evalLabels, batchSize=kerasBatchSize)[1]
        HLDRResult = protectedModel.evaluate([corruptedTestX, corruptedTestX], evalLabels, batchSize=kerasBatchSize)[1]
        hlrgdResult = hlrgd.evaluate([corruptedTestX, corruptedTestX], evalLabels, batchSize=kerasBatchSize)[1]
        fimResult = eigenRegModel.evaluate([corruptedTestX, corruptedTestX],evalLabels, batchSize=kerasBatchSize)[1]

        #collate reattackResults
        noiseAccuracy['undefended'].append(unprotectedResult)
        noiseAccuracy['hldr'].append(HLDRResult)
        noiseAccuracy['hlrgd'].append(hlrgdResult)
        noiseAccuracy['fim'].append(fimResult)

        if (noisyTraining):
            awgnResult = awgnModel.evaluate([corruptedTestX, corruptedTestX], evalLabels, batchSize=kerasBatchSize)[1]
            noiseAccuracy['agn'].append(awgnResult)

    # if we need to, recalculate the noisy data
    if (noisePowers[-1] != adversarialMAEBudget):
        corruptedTestX = testX[:numberOfAdvSamplesTest, :, :, :] + np.random.normal(0, scale=np.sqrt(adversarialMAEBudget), size=(np.min([numberOfAdvSamplesTest, testX.shape[0]]), testX.shape[1], testX.shape[2], testX.shape[3]))
    # calculate awgn induced perturbations (only for the largest budget, which is currently curNoisePower
    curAdversarialPreds = protectedModel.hiddenModel.predict([corruptedTestX[:numberOfAdvSamplesTest, :, :, :], corruptedTestX[:numberOfAdvSamplesTest, :, :, :]])
    curAdversarialPredsHLRGD = hlrgd.ourVGG.hiddenModel.predict([corruptedTestX[:numberOfAdvSamplesTest, :, :, :], corruptedTestX[:numberOfAdvSamplesTest, :, :, :]])
    curAdversarialPredsUndef = unprotectedModel.hiddenModel.predict([corruptedTestX[:numberOfAdvSamplesTest, :, :, :], corruptedTestX[:numberOfAdvSamplesTest, :, :, :]])
    curAdversarialPredsFIM = eigenRegModel.hiddenModel.predict([corruptedTestX[:numberOfAdvSamplesTest, :, :, :], corruptedTestX[:numberOfAdvSamplesTest, :, :, :]])
    if (noisyTraining):
        curAdversarialPredsAGN = awgnModel.hiddenModel.predict([corruptedTestX[:numberOfAdvSamplesTest, :, :, :], corruptedTestX[:numberOfAdvSamplesTest, :, :, :]])
    layerWiseAGNHLDRMAEs, layerWiseAGNHLRGDMAEs, layerWiseAGNUndefMAEs, layerWiseAGNFIMMAEs, layerWiseAGNAGNMAEs = [], [], [], [], []
    for curLayer in list(range(len(curAdversarialPreds)))[1::2]:
        curAdversarialMAEs = []
        HLDRMAE = np.mean((np.sum(np.abs(curAdversarialPreds[curLayer] - curAdversarialPreds[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPreds[curLayer+1]), axis=(1,2,3))), axis=0)
        hlrgdMAE = np.mean((np.sum(np.abs(curAdversarialPredsHLRGD[curLayer] - curAdversarialPredsHLRGD[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsHLRGD[curLayer+1]), axis=(1,2,3))), axis=0)
        undefMAE = np.mean((np.sum(np.abs(curAdversarialPredsUndef[curLayer] - curAdversarialPredsUndef[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsUndef[curLayer+1]), axis=(1,2,3))), axis=0)
        fimMAE = np.mean((np.sum(np.abs(curAdversarialPredsFIM[curLayer] - curAdversarialPredsFIM[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsFIM[curLayer+1]), axis=(1,2,3))), axis=0)
        layerWiseAGNHLDRMAEs.append(HLDRMAE)
        layerWiseAGNHLRGDMAEs.append(hlrgdMAE)
        layerWiseAGNUndefMAEs.append(undefMAE)
        layerWiseAGNFIMMAEs.append(fimMAE)
        if (noisyTraining):
            awgnMAE = np.mean((np.sum(np.abs(curAdversarialPredsAGN[curLayer] - curAdversarialPredsAGN[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsAGN[curLayer+1]), axis=(1,2,3))), axis=0)
            layerWiseAGNAGNMAEs.append(awgnMAE)
########################################################################################################################

########################################################################################################################
    #plot AGN perturbations
    plt.figure()
    numberOfLayers = len(layerWiseAGNUndefMAEs)
    curLevel =curNoisePower
    curCVIt = cvIndex
    plt.plot(range(numberOfLayers), layerWiseAGNUndefMAEs, marker='o', linestyle='-.', label=r'undefended $\rho$=%s p %s'%(str(curLevel*255.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseAGNHLDRMAEs, marker='*', linestyle='-', label=r'HLDR $\rho$=%s p %s'%(str(curLevel*255.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseAGNHLRGDMAEs, marker='x', label=r'hlrgd $\rho$=%s p %s' %(str(curLevel*255.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseAGNFIMMAEs, marker='d', label=r'fim $\rho$=%s p %s' %(str(curLevel*255.), str(curCVIt)))
    if (noisyTraining):
        plt.plot(range(numberOfLayers), layerWiseAGNAGNMAEs, marker='v', label=r'awgn $\rho$=%s p %s' %(str(curLevel*255.), str(curCVIt)))
    plt.ylabel(r'layer-wise perturbation: mean of ($\frac{\vert h_i(x)-h_i(x_a)\vert}{\vert h_i(x) \vert} $)')
    plt.xlabel('hidden layer index ($i$)')
    plt.legend()
    plt.savefig('layerwiseAGNPerturbations%s_.png'%str(cvIndex))
    # #plt.show()

    #if we need to, recalculate the attacks
    if (powersAcc[-1] != adversarialMAEBudget):
        testingAttacks = testX[:numberOfAdvSamplesTest, :, :, :] + adversarialMAEBudget*fgsmData['test']
    #calculate adversarial perturbations (only for the largest budget, which is currently curNoisePower
    curAdversarialPreds = protectedModel.hiddenModel.predict([testX[:numberOfAdvSamplesTest, :, :, :], testingAttacks[:numberOfAdvSamplesTest, :, :, :]])
    curAdversarialPredsHLRGD = hlrgd.ourVGG.hiddenModel.predict([testX[:numberOfAdvSamplesTest, :, :, :], testingAttacks[:numberOfAdvSamplesTest, :, :, :]])
    curAdversarialPredsUndef = unprotectedModel.hiddenModel.predict([testX[:numberOfAdvSamplesTest, :, :, :], testingAttacks[:numberOfAdvSamplesTest, :, :, :]])
    curAdversarialPredsFIM = eigenRegModel.hiddenModel.predict([testX[:numberOfAdvSamplesTest, :, :, :], testingAttacks[:numberOfAdvSamplesTest, :, :, :]])
    if (noisyTraining):
        curAdversarialPredsAGN = awgnModel.hiddenModel.predict([testX[:numberOfAdvSamplesTest, :, :, :], testingAttacks[:numberOfAdvSamplesTest, :, :, :]])
    layerWiseHLDRMAEs, layerWiseHLRGDMAEs, layerWiseUndefMAEs, layerWiseFIMMAEs, layerWiseAGNMAEs = [], [], [], [], []
    for curLayer in list(range(len(curAdversarialPreds)))[1::2]:
        curAdversarialMAEs = []
        HLDRMAE = np.mean((np.sum(np.abs(curAdversarialPreds[curLayer] - curAdversarialPreds[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPreds[curLayer+1]), axis=(1,2,3))), axis=0)
        hlrgdMAE = np.mean((np.sum(np.abs(curAdversarialPredsHLRGD[curLayer] - curAdversarialPredsHLRGD[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsHLRGD[curLayer+1]), axis=(1,2,3))), axis=0)
        undefMAE = np.mean((np.sum(np.abs(curAdversarialPredsUndef[curLayer] - curAdversarialPredsUndef[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsUndef[curLayer+1]), axis=(1,2,3))), axis=0)
        fimMAE = np.mean((np.sum(np.abs(curAdversarialPredsFIM[curLayer] - curAdversarialPredsFIM[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsFIM[curLayer+1]), axis=(1,2,3))), axis=0)
        layerWiseHLDRMAEs.append(HLDRMAE)
        layerWiseHLRGDMAEs.append(hlrgdMAE)
        layerWiseUndefMAEs.append(undefMAE)
        layerWiseFIMMAEs.append(fimMAE)
        if (noisyTraining):
            layerWiseAGNMAE = np.mean((np.sum(np.abs(curAdversarialPredsAGN[curLayer] - curAdversarialPredsAGN[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsAGN[curLayer+1]), axis=(1,2,3))), axis=0)
            layerWiseAGNMAEs.append(layerWiseAGNMAE)

    #plot adversarial perturbations
    plt.figure()
    curLevel =curNoisePower
    curCVIt = cvIndex
    plt.plot(range(numberOfLayers), layerWiseUndefMAEs, marker='o', linestyle='-.', label=r'undefended $\epsilon$=%s p %s'%(str(curLevel*255.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseHLDRMAEs, marker='*', linestyle='-', label=r'HLDR $\epsilon$=%s p %s'%(str(curLevel*255.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseHLRGDMAEs, marker='x', label=r'hlrgd $\epsilon$=%s p %s' %(str(curLevel*255.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseFIMMAEs, marker='d', label=r'fim $\epsilon$=%s p %s' %(str(curLevel*255.), str(curCVIt)))
    if (noisyTraining):
        plt.plot(range(numberOfLayers), layerWiseAGNMAEs, marker='v', label=r'fim $\epsilon$=%s p %s' %(str(curLevel*255.), str(curCVIt)))
    plt.ylabel(r'layer-wise perturbation: mean of ($\frac{\vert h_i(x)-h_i(x_a)\vert}{\vert h_i(x) \vert} $)')
    plt.xlabel('hidden layer index ($i$)')
    plt.legend()
    plt.savefig('layerwisePerturbations%s_.png'%str(cvIndex))
    #plt.show()

    # store results
    crossValResults['Acc'][cvIndex] = advAccuracy
    crossValResults['noiseAcc'][cvIndex] = noiseAccuracy
    crossValResults['averageLayerwiseUndefMAEs'][cvIndex] = layerWiseUndefMAEs
    crossValResults['averageLayerwiseHLDRMAEs'][cvIndex] = layerWiseHLDRMAEs
    crossValResults['averageLayerwiseHLRGDMAEs'][cvIndex] = layerWiseHLRGDMAEs
    crossValResults['averageLayerwiseFIMMAEs'][cvIndex] = layerWiseFIMMAEs
    crossValResults['averageLayerwiseAGNUndefMAEs'][cvIndex] = layerWiseAGNUndefMAEs
    crossValResults['averageLayerwiseAGNHLDRMAEs'][cvIndex] = layerWiseAGNHLDRMAEs
    crossValResults['averageLayerwiseAGNHLRGDMAEs'][cvIndex] = layerWiseAGNHLRGDMAEs
    crossValResults['averageLayerwiseAGNFIMMAEs'][cvIndex] = layerWiseAGNFIMMAEs
    if (noisyTraining):
        crossValResults['averageLayerwiseAGNMAEs'][cvIndex] = layerWiseAGNMAEs
        crossValResults['averageLayerwiseAGNAGNMAEs'][cvIndex] = layerWiseAGNAGNMAEs

    #pickle the results
    pickle.dump(crossValResults, open('crossValResults'+dataset+'.pickle', 'wb'))

    #convert acc results into numpy arrays for easy arithmetic
    advAccuracy['undefended'] = np.array(advAccuracy['undefended'])
    advAccuracy['hldr'] = np.array(advAccuracy['hldr'])
    advAccuracy['hlrgd'] = np.array(advAccuracy['hlrgd'])
    advAccuracy['fim'] = np.array(advAccuracy['fim'])
    noiseAccuracy['undefended'] = np.array(noiseAccuracy['undefended'])
    noiseAccuracy['hldr'] = np.array(noiseAccuracy['hldr'])
    noiseAccuracy['hlrgd'] = np.array(noiseAccuracy['hlrgd'])
    noiseAccuracy['fim'] = np.array(noiseAccuracy['fim'])
    if (noisyTraining):
        noiseAccuracy['agn'] = np.array(noiseAccuracy['agn'])

    #plot noise acc results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('AGN accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('noise power (p)')
    noisePowersPlot = copy.copy(noisePowers)
    noisePowersPlot.insert(0, 0)
    plt.plot(noisePowersPlot, noiseAccuracy['hldr'], label='HLDR')
    plt.plot(noisePowersPlot, noiseAccuracy['hlrgd'], label='HLRGD')
    plt.plot(noisePowersPlot, noiseAccuracy['fim'], label='FIM')
    if (noisyTraining):
        plt.plot(noisePowersPlot, noiseAccuracy['agn'], label='awgn')
    plt.legend()
    plt.savefig('acc_vs_budget_cv_%sAGN.png' % str(cvIndex))
    texts = []
    for i in range(len(noiseAccuracy['undefended'])):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['undefended'][i]))))
    for i in range(len(noiseAccuracy['hldr'])):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['hldr'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['hldr'][i]))))
    for i in range(len(noiseAccuracy['hlrgd'])):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['hlrgd'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['hlrgd'][i]))))
    for i in range(len(noiseAccuracy['fim'])):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['fim'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['fim'][i]))))
    if (noisyTraining):
        for i in range(len(noiseAccuracy['agn'])):
                texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['agn'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['agn'][i]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('acc_vs_budget_cv_%sAGN.png'%str(cvIndex))
    #plt.show()


    #noise marginal improvement plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('AGN improvement')
    plt.ylabel('marginal improvement in accuracy')
    plt.xlabel('noise power (p)')
    plt.plot(noisePowersPlot, noiseAccuracy['hldr']-noiseAccuracy['undefended'], label='HLDR')
    plt.plot(noisePowersPlot, noiseAccuracy['hlrgd']-noiseAccuracy['undefended'], label='HLRGD')
    plt.plot(noisePowersPlot, noiseAccuracy['fim'] - noiseAccuracy['undefended'], label='FIM')
    if (noisyTraining):
        plt.plot(noisePowersPlot, noiseAccuracy['agn'] - noiseAccuracy['undefended'], label='AGN')
    plt.legend()
    plt.savefig('marg_improv_vs_budget_%sAGN.png' % str(cvIndex))
    texts = []
    # for i in range(len(advAccuracy['undefended'])):
    #     texts.append(plt.text(powersAccAx[i], advAccuracy['undefended'][i], '(%s, %s)'%(str(powersAcc[i]), str(advAccuracy['undefended'][i]))))
    for i in range(len(noiseAccuracy['hldr'])):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['hldr'][i]-noiseAccuracy['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['hldr'][i]))))
    for i in range(len(noiseAccuracy['hlrgd'])):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['hlrgd'][i]-noiseAccuracy['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['hlrgd'][i]))))
    for i in range(len(noiseAccuracy['fim'])):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['fim'][i]-noiseAccuracy['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['fim'][i]))))
    for i in range(len(noiseAccuracy['agn'])):
            texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['agn'][i]-noiseAccuracy['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['agn'][i]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('marg_improv_vs_budget_%sAGN_withLabels.png'%str(cvIndex))
    #plt.show()


    #plot acc results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('adversarial accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('adversarial budget')
    powersAccPlot = copy.copy(powersAcc)
    powersAccPlot.insert(0, 0)
    powersAccAx = np.array(powersAccPlot)*(255.)
    plt.plot(powersAccAx, advAccuracy['hldr'], label='HLDR')
    plt.plot(powersAccAx, advAccuracy['hlrgd'], label='HLRGD')
    plt.plot(powersAccAx, advAccuracy['fim'], label='FIM')
    if (noisyTraining):
        plt.plot(powersAccAx, advAccuracy['agn'], label='AGN')
    plt.legend()
    plt.savefig('acc_vs_budget_cv_%s.png' % str(cvIndex))
    texts = []
    for i in range(len(advAccuracy['undefended'])):
        texts.append(plt.text(powersAccAx[i], advAccuracy['undefended'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(advAccuracy['undefended'][i]))))
    for i in range(len(advAccuracy['hldr'])):
        texts.append(plt.text(powersAccAx[i], advAccuracy['hldr'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(advAccuracy['hldr'][i]))))
    for i in range(len(advAccuracy['hlrgd'])):
        texts.append(plt.text(powersAccAx[i], advAccuracy['hlrgd'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(advAccuracy['hlrgd'][i]))))
    for i in range(len(advAccuracy['fim'])):
        texts.append(plt.text(powersAccAx[i], advAccuracy['fim'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(advAccuracy['fim'][i]))))
    if (noisyTraining):
        for i in range(len(advAccuracy['agn'])):
            texts.append(plt.text(powersAccAx[i], advAccuracy['agn'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(advAccuracy['agn'][i]))))
    plt.legend()
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('acc_vs_budget_cv_%s_withLabels.png'%str(cvIndex))
    #plt.show()


    #marginal improvement plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('adversarial accuracy')
    plt.ylabel('marginal improvement in accuracy')
    plt.xlabel('adversarial budget (p)')
    plt.plot(np.array(powersAccPlot)*(255.), advAccuracy['hldr']-advAccuracy['undefended'], label='HLDR')
    plt.plot(np.array(powersAccPlot)*(255.), advAccuracy['hlrgd']-advAccuracy['undefended'], label='HLRGD')
    plt.plot(np.array(powersAccPlot) * (255), advAccuracy['fim'] - advAccuracy['undefended'], label='FIM')
    if (noisyTraining):
        plt.plot(np.array(powersAccPlot) * (255), advAccuracy['agn'] - advAccuracy['undefended'], label='AGN')
    plt.legend()
    plt.savefig('marg_improv_vs_budget_%s.png' % str(cvIndex))
    texts = []
    # for i in range(len(advAccuracy['undefended'])):
    #     texts.append(plt.text(powersAccAx[i], advAccuracy['undefended'][i], '(%s, %s)'%(str(powersAcc[i]), str(advAccuracy['undefended'][i]))))
    for i in range(len(advAccuracy['hldr'])):
        texts.append(plt.text(powersAccAx[i], advAccuracy['hldr'][i]-advAccuracy['undefended'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(advAccuracy['hldr'][i]))))
    for i in range(len(advAccuracy['hlrgd'])):
        texts.append(plt.text(powersAccAx[i], advAccuracy['hlrgd'][i]-advAccuracy['undefended'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(advAccuracy['hlrgd'][i]))))
    for i in range(len(advAccuracy['fim'])):
        texts.append(plt.text(powersAccAx[i], advAccuracy['fim'][i]-advAccuracy['undefended'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(advAccuracy['fim'][i]))))
    if (noisyTraining):
        for i in range(len(advAccuracy['agn'])):
            texts.append(plt.text(powersAccAx[i], advAccuracy['agn'][i]-advAccuracy['undefended'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(advAccuracy['agn'][i]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('marg_improv_vs_budget_%s_withLabels.png'%str(cvIndex))
    #plt.show()

    #delete old models and clear the backend so we don't fill up the gpu memory
    if (cvIndex < numberOfReplications-1):
        hlrSGD, protSGD = hlrgd.getOptimizer(), protectedModel.getOptimizer()
        del hlrSGD
        del protSGD
        del hlrgd
        del protectedModel
        del trainX
        del testX
        for curKey in list(fgsmData.keys()):
            del fgsmData[curKey]
        tf.keras.backend.clear_session()
    cvIndex += 1
########################################################################################################################
