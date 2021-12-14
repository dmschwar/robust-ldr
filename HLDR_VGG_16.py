########################################################################################################################
#
# Author: David Schwartz, June, 9, 2020
#
# This file implements our contribution, the HLDR defense method paired with VGG.
########################################################################################################################

########################################################################################################################
import sys
import scipy
import pickle
from itertools import chain
from sklearn.metrics.cluster import adjusted_mutual_info_score, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, trange
from sklearn.model_selection import StratifiedShuffleSplit


from sklearn.utils import _safe_indexing as safe_indexing, indexable
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, TimeDistributed, Conv1D, BatchNormalization

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import os
import cv2
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras import regularizers, losses, utils
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import time
import re, random, collections
from collections import defaultdict
from numpy import linalg as LA
import itertools
########################################################################################################################


########################################################################################################################
defaultPatience = 25
defaultLossThreshold = 0.00001
defaultDropoutRate = 0

class HLDRVGG(object):
    def __init__(self, input_dimension, output_dimension, number_of_classes=2, optimizer=None, dual_outputs=False,
        loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=defaultDropoutRate, reg='HLDR',
        max_relu_bound=None, adv_penalty=0.01, unprotected=False, freezeWeights=False, verbose=False):

        self.buildModel(input_dimension, output_dimension, number_of_classes=number_of_classes, dual_outputs=dual_outputs,
                        loss_threshold=loss_threshold, patience=patience, dropout_rate=dropout_rate,
                        max_relu_bound=max_relu_bound, adv_penalty=adv_penalty, unprotected=unprotected,
                        optimizer=optimizer, reg=reg, freezeWeights=freezeWeights, verbose=verbose)


    def buildModel(self, input_dimension, output_dimension, number_of_classes=2, optimizer=None, dual_outputs=False,
        loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=defaultDropoutRate, 
        max_relu_bound=None, adv_penalty=0.01, unprotected=False, reg='HLDR', freezeWeights=False, verbose=False):
        self.input_dimension, self.output_dimension = input_dimension, np.copy(output_dimension)
        self.advPenalty = np.copy(adv_penalty)

        self.loss_threshold, self.number_of_classes = np.copy(loss_threshold), np.copy(number_of_classes)
        self.dropoutRate, self.max_relu_bound = np.copy(dropout_rate), np.copy(max_relu_bound)
        self.patience = np.copy(patience)
        self.dualOutputs = dual_outputs
        self.image_size = 32
        self.num_channels = 3
        self.num_labels = np.copy(number_of_classes)
        self.penaltyCoeff = np.copy(adv_penalty)

        #decide an activation function
        self.chosenActivation = "relu" if max_relu_bound is not None else "tanh"

        if (verbose):
            print("input dimension: %s"%str(self.input_dimension))

        # define input layer
        self.inputLayer = layers.Input(shape=self.input_dimension)
        previousLayer = self.inputLayer

        #define the adversarial main input layer (only used in training)
        self.advInputLayer = layers.Input(shape=self.input_dimension)
        previousAdvLayer = self.advInputLayer

        #define the hidden layers
        self.hiddenLayers = dict()
        self.hiddenAdvLayers = dict()
        self.poolingLayers = dict()

        #define hidden layer outputs
        self.hiddenModelOutputs, self.hiddenAdvModelOutputs = dict(), dict()

        #layer 0
        self.hiddenLayers[0] = layers.Conv2D(64, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[0] = self.hiddenLayers[0]
        previousLayer = self.hiddenLayers[0](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[0](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[0] = previousLayer
        self.hiddenAdvModelOutputs[0] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 1
        self.hiddenLayers[1] = layers.Conv2D(64, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[1] = self.hiddenLayers[1]
        previousLayer = self.hiddenLayers[1](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[1](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[1] = previousLayer
        self.hiddenAdvModelOutputs[1] = previousAdvLayer

        #pooling
        self.poolingLayers[0] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[0](previousLayer)
        previousAdvLayer = self.poolingLayers[0](previousAdvLayer)

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 2
        self.hiddenLayers[2] = layers.Conv2D(128, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[2] = self.hiddenLayers[2]
        previousLayer = self.hiddenLayers[2](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[2](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[2] = previousLayer
        self.hiddenAdvModelOutputs[2] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 3
        self.hiddenLayers[3] = layers.Conv2D(128, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[3] = self.hiddenLayers[3]
        previousLayer = self.hiddenLayers[3](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[3](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[3] = previousLayer
        self.hiddenAdvModelOutputs[3] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        # pooling
        self.poolingLayers[1] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[1](previousLayer)
        previousAdvLayer = self.poolingLayers[1](previousAdvLayer)


        #layer 4
        self.hiddenLayers[4] = layers.Conv2D(256, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[4] = self.hiddenLayers[4]
        previousLayer = self.hiddenLayers[4](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[4](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[4] = previousLayer
        self.hiddenAdvModelOutputs[4] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 5
        self.hiddenLayers[5] = layers.Conv2D(256, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[5] = self.hiddenLayers[5]
        previousLayer = self.hiddenLayers[5](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[5](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[5] = previousLayer
        self.hiddenAdvModelOutputs[5] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 6
        self.hiddenLayers[6] = layers.Conv2D(256, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[6] = self.hiddenLayers[6]
        previousLayer = self.hiddenLayers[6](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[6](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[6] = previousLayer
        self.hiddenAdvModelOutputs[6] = previousAdvLayer

        # pooling
        self.poolingLayers[2] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[2](previousLayer)
        previousAdvLayer = self.poolingLayers[2](previousAdvLayer)

        #layer 7
        self.hiddenLayers[7] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[7] = self.hiddenLayers[7]
        previousLayer = self.hiddenLayers[7](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[7](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[7] = previousLayer
        self.hiddenAdvModelOutputs[7] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 8
        self.hiddenLayers[8] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[8] = self.hiddenLayers[8]
        previousLayer = self.hiddenLayers[8](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[8](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[8] = previousLayer
        self.hiddenAdvModelOutputs[8] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 9
        self.hiddenLayers[9] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[9] = self.hiddenLayers[9]
        previousLayer = self.hiddenLayers[9](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[9](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[9] = previousLayer
        self.hiddenAdvModelOutputs[9] = previousAdvLayer


        #pooling
        self.poolingLayers[3] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[3](previousLayer)
        previousAdvLayer = self.poolingLayers[3](previousAdvLayer)

        #layer 10
        self.hiddenLayers[10] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[10] = self.hiddenLayers[10]
        previousLayer = self.hiddenLayers[10](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[10](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[10] = previousLayer
        self.hiddenAdvModelOutputs[10] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 11
        self.hiddenLayers[11] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[11] = self.hiddenLayers[11]
        previousLayer = self.hiddenLayers[11](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[11](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[11] = previousLayer
        self.hiddenAdvModelOutputs[11] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 12
        self.hiddenLayers[12] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[12] = self.hiddenLayers[12]
        previousLayer = self.hiddenLayers[12](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[12](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[12] = previousLayer
        self.hiddenAdvModelOutputs[12] = previousAdvLayer

        # pooling
        self.poolingLayers[4] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[4](previousLayer)
        previousAdvLayer = self.poolingLayers[4](previousAdvLayer)

        #dense layers
        previousLayer = layers.Flatten()(previousLayer)
        previousAdvLayer = layers.Flatten()(previousAdvLayer)

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        self.penultimateDenseLayer = layers.Dense(512, activation=self.chosenActivation, kernel_regularizer=regularizers.l2(5e-5))
        self.hiddenLayers[13] = self.penultimateDenseLayer
        self.hiddenAdvLayers[13] = self.penultimateDenseLayer
        previousLayer = self.penultimateDenseLayer(previousLayer)
        previousAdvLayer = self.penultimateDenseLayer(previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[13] = previousLayer
        self.hiddenAdvModelOutputs[13] = previousAdvLayer
        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #add the output layer
        #size constrained by dimensionality of inputs
        # self.logitsLayer = layers.Dense(output_dimension, activation=None, name='logitsLayer')
        # self.penultimateLayer = self.logitsActivity = self.logitsLayer(previousLayer)
        # self.penultimateAdvLayer = advLogitsActivity = self.logitsLayer(previousAdvLayer)
        self.outputLayer = layers.Dense(output_dimension, activation='softmax', name='outputlayer')#layers.Activation('softmax')
        self.outputActivity = self.outputLayer(previousLayer)
        self.advOutputActivity = self.outputLayer(previousAdvLayer)

        #set up the logits layer (not just breaking apart the outputlayer because we want to be able to read in old pretrained models, so we'll just invert for this
        #softmax^-1 (X) at coordinate i = log(X_i) + log(\sum_j exp(X_j))
        self.logitsActivity = K.log(self.outputActivity) #+ K.log(K.sum(K.exp(self.outputActivity)))
        self.advLogitsActivity = K.log(self.advOutputActivity) #+ K.log(K.sum(K.exp(self.advOutputActivity)))
        self.hiddenModelOutputs[14] = self.logitsActivity
        self.hiddenAdvLayers[14] = self.advLogitsActivity


        # # setup the models with which we can see states of hidden layers
        numberOfHiddenLayers = len(self.hiddenLayers)



        #collect adversarial projections and benign projections

        benProjs = layers.concatenate([layers.Flatten()(self.hiddenModelOutputs[curLayer]) for curLayer in range(numberOfHiddenLayers)])
        advProjs = layers.concatenate([layers.Flatten()(self.hiddenAdvModelOutputs[curLayer]) for curLayer in range(numberOfHiddenLayers)])
        self.benProjs, self.advProjs = benProjs, advProjs

        #define our custom loss function depending on how the intializer wants to regularize (i.e., the "reg" argument)
        #this is cross_entropy + \sum_layers(abs(benign_projection-adv_projection))
        self.unprotected = unprotected
        self.reg = reg
        if (reg == 'HLDR'):
            if (not unprotected):
                def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(tf.cast(K.sum(K.abs(benProjs)), tf.float32))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(tf.cast(tf.shape(benProjs)[0], tf.float32))
                    return customLoss
            else:#if we are using an unprotected model, don't force the  machine to calculate this too
                def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        return K.categorical_crossentropy(y_true, y_pred)
                    return customLoss
        elif (reg == 'HLRDivlayer'):
            if (not unprotected):
                # numerators = K.abs(layerWiseBenProjs - layerWiseAdvProjs)
                summands = [K.sum(K.abs(self.hiddenLayers[curLayer].output - self.hiddenAdvLayers[curLayer].output))/K.sum(K.abs(self.hiddenLayers[curLayer].output)) for curLayer in range(numberOfHiddenLayers)]
                def customLossWrapper(summands, penaltyCoeff = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(summands)
                    return customLoss
            else:#if we are using an unprotected model, don't force the  machine to calculate this too
                def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        return K.categorical_crossentropy(y_true, y_pred)
                    return customLoss
        elif (reg ==  'FIM'):
                # dS = tf.gradients(self.outputLayer, self.inputLayer)
                # dS_2 = tf.matmul(dS, tf.reshape(dS, (dS.shape[1], dS.shape[0])))
                # eigs = tf.linalg.eigvals(dS_2)
                ps = tf.divide(tf.ones(shape=(tf.shape(self.outputActivity))), tf.where(self.outputActivity > 0, self.outputActivity, 1e16*tf.ones_like(self.outputActivity)))
                def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(ps)
                    return customLoss
        elif (reg ==  'logEigen'):
            ps = tf.divide(tf.ones(shape=(tf.shape(self.outputActivity))), tf.ones_like(self.outputActivity)-tf.math.log(tf.where(self.outputActivity > 0, self.outputActivity, 1e16*tf.ones_like(self.outputActivity))))
            def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                def customLoss(y_true, y_pred):
                    return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(ps)
                return customLoss
        elif (reg ==  'logEigenlogits'):
            ps = tf.divide(tf.ones(shape=(tf.shape(self.logitsActivity))), tf.ones_like(self.logitsActivity)+tf.math.log(tf.where(self.logitsActivity > 0, self.logitsActivity, 1e16*tf.ones_like(self.logitsActivity))))
            def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                def customLoss(y_true, y_pred):
                    return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(ps)
                return customLoss
        elif (reg ==  'logitFIM'):
            ps = tf.divide(tf.ones(shape=(tf.shape(self.logitsActivity))), tf.where(self.logitsActivity > 0, self.logitsActivity, 1e16*tf.ones_like(self.logitsActivity)))
            def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                def customLoss(y_true, y_pred):
                    return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(ps)
                return customLoss
        else:
            def customLossWrapper(benProjs, advProjs, penaltyCoeff=self.penaltyCoeff):
                def customLoss(y_true, y_pred):
                    return K.categorical_crossentropy(y_true, y_pred)
                return customLoss
        
        self.sgd = tf.keras.optimizers.Nadam()#Adadelta(learning_rate=self.learning_rate)
        self.reduceLR = None

        #set up data augmentation
        self.generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)

        #convert self.hiddenAdvLayers to a list for the model compilation, ascending order of keys is order of layers
        #outputsList is a list of outputs of the model constructed so that the first entry is the true output (ie prediction) layer
        #and each subsequent (i, i+1)th entries are the pair of hiddenAdvLayer, hiddenBenignLayer activations
        #this is going to be useful for calculating the MAE between benignly and adversarially induced hidden states
        outputsList = [self.outputActivity]
        oneSideOutputsList = [self.outputActivity]
        for curHiddenLayer in range(len(self.hiddenAdvModelOutputs))[:-1]:
            oneSideOutputsList.append(self.hiddenModelOutputs[curHiddenLayer])
            outputsList.append(self.hiddenAdvModelOutputs[curHiddenLayer])
            outputsList.append(self.hiddenModelOutputs[curHiddenLayer])

        mainOutputList = [self.outputActivity]
        if (self.dualOutputs):
            mainOutputList.append(self.logitsActivity)
            #mainOutputList.append(self.advOutputActivity)
            #mainOutputList.append(self.advLogitsActivity)

        # instantiate and compile the model
        self.customLossWrapper = customLossWrapper
        self.model = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=mainOutputList, name='HLDR_vgg_16')
                #if we want to use this as a frozen model
        if (freezeWeights):
            for curWeights in range(len(self.model.layers)):
                self.model.layers[curWeights].trainable = False
        if (reg == 'HLRlayer'):
            self.model.compile(loss=customLossWrapper(summands, self.penaltyCoeff), metrics=['acc'], optimizer=self.sgd)
        else:
            self.model.compile(loss=customLossWrapper(benProjs, advProjs, self.penaltyCoeff), metrics=['acc'], optimizer=self.sgd)

        #setup the models with which we can see states of hidden layers
        self.hiddenModel = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=outputsList, name='hidden_HLDR_vgg_16')
        self.hiddenOneSideModel = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=oneSideOutputsList, name='hidden_oneside_HLDR_vgg_16')
        self.hiddenJointLatentModel = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=[benProjs], name='hiddenJointLatentModel')
        self.logitModel = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=[self.logitsActivity], name='hiddenLogitModel')
        # double check weight trainability bug
        allVars = self.model.variables
        trainableVars = self.model.trainable_variables
        allVarNames = [self.model.variables[i].name for i in range(len(self.model.variables))]
        trainableVarNames = [self.model.trainable_variables[i].name for i in range(len(self.model.trainable_variables))]
        nonTrainableVars = np.setdiff1d(allVarNames, trainableVarNames)

        if (verbose):
            self.model.summary()
            if (len(nonTrainableVars) > 0):
                print('the following variables are set to non-trainable; ensure that this is correct before publishing!!!!')
            print(nonTrainableVars)

        #set data statistics to default values
        self.mean = 0
        self.stddev = 1

    #this routine is used to collect statistics on training data, as well as to preprocess the training data by normalizing
    #i.e. centering and dividing by standard deviation
    def normalize(self, inputData, storeStats=False):
        if (storeStats):
            self.mean = np.mean(inputData)
            self.stddev = np.std(inputData)
        outputData = (inputData-self.mean)/(self.stddev + 0.0000001)
        return outputData

    # routine to get a pointer to the optimizer of this model
    def getOptimizer(self):
        return self.sgd
    #
    def getVGGWeights(self):
        return self.model.get_weights().copy()

    def getParameterCount(self):
        return self.model.count_params()

    # handle data augmentation with multiple inputs (example found on https://stackoverflow.com/questions/49404993/keras-how-to-use-fit-generator-with-multiple-inputs
    #so thanks to loannis and Julian
    def multiInputDataGenerator(self, X1, X2, Y, batch_size):
        genX1 = self.generator.flow(X1, Y, batch_size=batch_size)
        genX2 = self.generator.flow(X2, Y, batch_size=batch_size)

        while True:
            X1g = genX1.next()
            X2g = genX2.next()
            yield [X1g[0], X2g[0]], X1g[1]

    #adversarial order parameter tells us if we're doing adversarial training, so we know if we should normalize to the first or second argument
    def train(self, inputTrainingData, trainingTargets, inputValidationData, validationTargets, training_epochs=1, 
                normed=False, monitor='val_loss', patience=defaultPatience,
                model_path=None, keras_batch_size=None, dataAugmentation=False, adversarialOrder=0):
        #if a path isn't provided by caller, just use the current time for restoring best weights from fit
        if (model_path is None):
            model_path = os.path.join('/tmp/models/', 'hlr_vgg16_'+str(int(round(time.time()*1000))))

        #if the data are not normalized, normalize them
        trainingData, validationData = [[],[]], [[],[]]
        if (not normed):
            #don't store stats from the adversarially attacked data
            if (adversarialOrder == 0):
                trainingData[0] = self.normalize(inputTrainingData[0], storeStats=True)
                trainingData[1] = self.normalize(inputTrainingData[1], storeStats=False)
            else:
                trainingData[1] = self.normalize(inputTrainingData[1], storeStats=True)
                trainingData[0] = self.normalize(inputTrainingData[0], storeStats=False)
            #also don't store stats from validation data
            validationData[0] = self.normalize(inputValidationData[0], storeStats=False)
            validationData[1] = self.normalize(inputValidationData[1], storeStats=False)
        else:
            trainingData[0] = inputTrainingData[0]
            trainingData[1] = inputTrainingData[1]
            validationData[0] = inputValidationData[0]
            validationData[1] = inputValidationData[1]

        #collect our callbacks
        earlyStopper = EarlyStopping(monitor=monitor, mode='min', patience=patience,
                                     verbose=1, min_delta=defaultLossThreshold)
        checkpoint = ModelCheckpoint(model_path, verbose=1, monitor=monitor, save_weights_only=True,
                                     save_best_only=True, mode='auto')
        callbackList = [earlyStopper, checkpoint]
        # history = self.model.fit(trainingData, trainingTargets, epochs=training_epochs, batch_size=keras_batch_size,
        #                          validation_split=validation_split, callbacks=[earlyStopper, self.reduce_lr])
        #handle data augmentation
        if (not dataAugmentation):
            # set up data augmentation
            self.generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                                samplewise_center=False,  # set each sample mean to 0
                                                featurewise_std_normalization=False,
                                                # divide inputs by std of the dataset
                                                samplewise_std_normalization=False,  # divide each input by its std
                                                zca_whitening=False,  # apply ZCA whitening
                                                # randomly shift images vertically (fraction of total height)
                                                horizontal_flip=False,# randomly flip images
                                                vertical_flip=False)
            self.generator.fit(trainingData[0])
            history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, keras_batch_size),
                                               steps_per_epoch=trainingData[0].shape[0] // keras_batch_size,
                                               epochs=training_epochs, validation_data=(validationData, validationTargets),
                                               callbacks=callbackList, verbose=1) #self.reduce_lr
        else:
            # set up data augmentation
            self.generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                                samplewise_center=False,  # set each sample mean to 0
                                                featurewise_std_normalization=False,
                                                # divide inputs by std of the dataset
                                                samplewise_std_normalization=False,  # divide each input by its std
                                                zca_whitening=False,  # apply ZCA whitening
                                                rotation_range=15,
                                                # randomly rotate images in the range (degrees, 0 to 180)
                                                width_shift_range=0.1,
                                                # randomly shift images horizontally (fraction of total width)
                                                height_shift_range=0.1,
                                                # randomly shift images vertically (fraction of total height)
                                                horizontal_flip=False,  # randomly flip images
                                                vertical_flip=False)
            self.generator.fit(trainingData[0])
            history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, keras_batch_size),
                                               steps_per_epoch=trainingData[0].shape[0] // keras_batch_size,
                                               epochs=training_epochs, validation_data=(validationData, validationTargets),
                                               callbacks=callbackList, verbose=1) #self.reduce_lr
        if (not np.isnan(history.history['loss']).any() and not np.isinf(history.history['loss']).any()):
	        self.model.load_weights(model_path)
        loss, acc = history.history['loss'], history.history['val_acc']
        return loss, acc, model_path

    def evaluate(self, inputData, targets, batchSize=None):
        evalData = [self.normalize(inputData[0], storeStats=False), self.normalize(inputData[1], storeStats=False)]
        fullEval = self.model.evaluate(evalData, targets, batch_size=batchSize)
        return fullEval

    # this method reads our models from the disk at the specified path + name of component of the class + h5
    # and reads  all other parameters from a pickle file
    def readModelFromDisk(self, pathToFile):

        #rebuild the model
        self.buildModel(self.input_dimension, self.output_dimension, self.number_of_classes,
                        loss_threshold=self.loss_threshold, patience=self.patience, dropout_rate=self.dropoutRate,
                        max_relu_bound=self.max_relu_bound, adv_penalty=self.advPenalty, unprotected=self.unprotected,
                        reg=self.reg, verbose=verbose)
        # set the vgg weights
        self.model.load_weights(pathToFile)
        # #read in the picklebox
        pickleBox = pickle.load(open(pathToFile+'_pickle', 'rb'))
        # # self.bottleneckLayer = pickleBox['bottleneckLayer']
        # # self.hiddenEncodingLayer = pickleBox['hiddenEncodingLayer']
        # # self.inputLayer = pickleBox['inputLayer']
        self.reg = pickleBox['reg']
        self.chosenActivation = pickleBox['chosenActivation']
        self.mean, self.std = pickleBox['scaleMean'], pickleBox['scaleSTD']
########################################################################################################################

########################################################################################################################
if __name__ == "__main__":
    #define test parameters
    # define parameters
    verbose = True
    testFraction = 0.5
    numberOfClasses = 10
    trainingSetSize = 5000
    numberOfAdvSamples = 1000
    trainingEpochs = 1
    powers = [0.05, 0.1, 0.25, 0.5, 1, 2]

    # set up data
    # input image dimensions
    img_rows, img_cols = 32, 32

    inputDimension = (32, 32, 1)
    # read in cifar data
    # split data between train and test sets
    (x_train, t_train), (x_test, t_test) = tf.keras.datasets.mnist.load_data()
    x_train_up, x_test_up = np.zeros((x_train.shape[0], 32, 32)), np.zeros((x_test.shape[0], 32, 32))
    # upscale data to 32,32 (same size as cifar)
    for i in range(x_train.shape[0]):
        x_train_up[i, :, :] = cv2.resize(x_train[i, :, :], (32, 32), interpolation=cv2.INTER_AREA)
    for i in range(x_test.shape[0]):
        x_test_up[i, :, :] = cv2.resize(x_test[i, :, :], (32, 32), interpolation=cv2.INTER_AREA)
    x_train = x_train_up
    x_test = x_test_up
    # x_train, x_test = vgg_preprocess_input(x_train), vgg_preprocess_input(x_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.
    # for mnist only: mean:0.13088295; std:0.30840662
    # for fashion mnist: mean:0.28654176; std:0.35317212

    # split cifarData by classes so we can distribute it
    task_permutation = []
    trainXs, trainTargets = [], []
    testXs, testTargets = [], []
    # split cifarData by classes so we can distribute it
    trainXs, trainTargets = [], []
    testXs, testTargets = [], []

    for t in range(numberOfClasses):
        # arrange training and validation data
        curClassIndicesTraining = np.where(t_train == t)[0]
        curXTrain = x_train[curClassIndicesTraining, :, :]
        trainXs.append(curXTrain)
        trainTargets.append(t * np.ones([curXTrain.shape[0], 1]))

        # arrange testing data
        curClassIndicesTesting = np.where(t_test == t)[0]
        curXTest = x_test[curClassIndicesTesting, :, :]
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

    trainX, testX, trainY, testY = model_selection.train_test_split(stackedData, stackedTargets,
                                                                    test_size=testFraction, random_state=42)
    trainX, testX, trainY, testY = trainX[:trainingSetSize, :, :], testX[:trainingSetSize, :, :], trainY[:trainingSetSize], testY[:trainingSetSize]
    print('beginning test')
    output_dimension = numberOfClasses
    ourModel = HLDRVGG(inputDimension, output_dimension, number_of_classes=numberOfClasses, adv_penalty=0.05,
                       loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=0.05,
                       max_relu_bound=1.1, reg='FIM', verbose=False)
    unprotectedModel = HLDRVGG([32, 32, 1], numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0.05,
                       loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=0.33,
                       max_relu_bound=1.1, verbose=False, unprotected=True)
    unprotectedModel.model.load_weights('unprotectedFashionMNIST.h5')

    print(ourModel.evaluate([np.expand_dims(testX, axis=3), np.expand_dims(np.zeros(testX.shape), axis=3)], tf.keras.utils.to_categorical(testY, numberOfClasses)))

    # attack this vgg to generate samples with fgsm
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

    # set up containers to store perturbations
    fgsmData = dict()
    fgsmData['train'] = np.zeros((numberOfAdvSamples, 32, 32, 1))
    fgsmData['test'] = np.zeros((numberOfAdvSamples, 32, 32, 1))
    # attack the unprotected model
    fgsmData['train'] = create_adversarial_pattern(np.expand_dims(trainX[:numberOfAdvSamples, :, :], axis=3),
                                                   tf.keras.utils.to_categorical(trainY[:numberOfAdvSamples],
                                                                                 num_classes=numberOfClasses))
    # fgsmData['test'] = create_adversarial_pattern(testX[:numberOfAdvSamples, :, :, :],
    #                                               tf.keras.utils.to_categorical(testY[:numberOfAdvSamples],
    #                                                                             num_classes=numberOfClasses))

    print("attack complete")

    #mock up the MAE vs layer plot for varying noise levels
    #first collect the perturbed MAE
    noiseLevels = [0.5, 1, 5, 10]
    averageLayerwiseMAEs = dict()
    for curNoiseLevel in noiseLevels:
        ourModel.train([np.expand_dims(trainX, axis=3), np.expand_dims(trainX, axis=3)],
                       tf.keras.utils.to_categorical(trainY, numberOfClasses),
                       [np.expand_dims(trainX, axis=3), np.expand_dims(trainX, axis=3)],
                       tf.keras.utils.to_categorical(trainY, numberOfClasses), training_epochs=trainingEpochs,
                       monitor='val_loss', patience=25, model_path=None, keras_batch_size=64, dataAugmentation=True)
        trainingAttacks = np.expand_dims(trainX[:numberOfAdvSamples, :, :], axis=3) + curNoiseLevel * K.eval(fgsmData['train'])
        testingAttacks = np.expand_dims(testX[:numberOfAdvSamples, :, :], axis=3) + curNoiseLevel * K.eval(fgsmData['test'])
        curAdversarialPreds = ourModel.hiddenModel.predict([trainingAttacks,
                                                            np.expand_dims(trainX[:numberOfAdvSamples, :, :], axis=3)])
        curLayerWiseMAEs = []
        for curLayer in list(range(len(curAdversarialPreds)))[1::2]:
            curLayerAEs = np.abs(curAdversarialPreds[curLayer]-curAdversarialPreds[curLayer+1])
            curAdversarialMAEs = []
            init = time.time()
            a = np.sum(np.sum(np.sum(
                np.sum(np.abs(curAdversarialPreds[curLayer] - curAdversarialPreds[curLayer + 1]), axis=0) /
                curAdversarialPreds[curLayer].shape[0], axis=0) / curAdversarialPreds[curLayer].shape[1], axis=0) /
                       curAdversarialPreds[curLayer].shape[2], axis=0) / curAdversarialPreds[curLayer].shape[3]
            atime = time.time() - init
            curLayerWiseMAEs.append(a)
            # print((a, atime))
            init = time.time()
            b = np.mean(np.mean(
                np.mean(np.mean(np.abs(curAdversarialPreds[curLayer] - curAdversarialPreds[curLayer + 1]), axis=0), axis=0),
                axis=0), axis=0)
            btime = time.time() - init
            print((curLayer+1)/2)
            print((a, b))
            print((atime, btime))
        averageLayerwiseMAEs[curNoiseLevel] = curLayerWiseMAEs


    #plot the maes
    plt.figure()
    numberOfLayers = np.round((len(curAdversarialPreds)-1)/2).astype(int)
    for curLevel in noiseLevels:
        plt.plot(range(numberOfLayers), averageLayerwiseMAEs[curLevel], label='sigma = %s'%str(curLevel))

    plt.legend()
    plt.savefig('layerwiseMAEs.png')
    plt.show()
########################################################################################################################
