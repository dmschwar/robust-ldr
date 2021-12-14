########################################################################################################################
#
# Author: David Schwartz, June, 9, 2020
#
# This file implements the Higher Level Representation Guided Denoiser (in tandem with VGG).
########################################################################################################################

########################################################################################################################
import sys
# if ('linux' not in sys.platform):
#     import pyBILT_common as pyBILT

# import infotheory
import scipy
import pickle
from itertools import chain
from sklearn.metrics.cluster import adjusted_mutual_info_score, mutual_info_score
# from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, trange
# import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras import Model
# from tensorflow.keras import regularizers
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.utils import safe_indexing, indexable
try:
    # See #1137: this allows compatibility for scikit-learn >= 0.24
    from sklearn.utils import safe_indexing, indexable
except ImportError:
    from sklearn.utils import _safe_indexing as safe_indexing, indexable
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, TimeDistributed, Conv1D, BatchNormalization
#tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
# from keras.callbacks import ModelCheckpoint#, TensorBoard
# from keras.callbacks import Callback
import os
import cv2
# from HLDR_VGG_16 import HLDRVGG
import warnings
from tensorflow.keras import regularizers, losses, utils
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from tqdm import tqdm, trange
import numpy as np
import time
import re, random, collections
from collections import defaultdict
from numpy import linalg as LA
# from ops import *
import itertools
from HLDR_VGG_16 import HLDRVGG as HLVGG
########################################################################################################################

########################################################################################################################
defaultPatience = 25
defaultLossThreshold = 0.00001
defaultDropoutRate = 0.5
numberOfFilters = [64, 128, 256, 256, 256, 256, 256, 128, 64]

class HLRGDVGG(object):
    def __init__(self, input_dimension, output_dimension, vggModelLocation, number_of_classes=2, optimizer=None,
                 loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=defaultDropoutRate,
                 max_relu_bound=None, adv_penalty=0.01, unprotected=False, verbose=False):

        self.buildModel(input_dimension=input_dimension, output_dimension=output_dimension , vggModelLocation=vggModelLocation, optimizer=optimizer,
                        number_of_classes=number_of_classes, loss_threshold=loss_threshold, patience=patience, dropout_rate=dropout_rate,
                        max_relu_bound=max_relu_bound, adv_penalty=adv_penalty, unprotected=unprotected, verbose=verbose)


    def buildModel(self, input_dimension, output_dimension, vggModelLocation, number_of_classes=2, optimizer=None,
                 loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=defaultDropoutRate,
                 max_relu_bound=None, adv_penalty=0.01, unprotected=False, verbose=False):
        #instantiate the vgg instance
        self.ourVGG = HLVGG(input_dimension=input_dimension, output_dimension=output_dimension, dual_outputs=True, number_of_classes=number_of_classes,
                            loss_threshold=loss_threshold, patience=patience, dropout_rate=0, freezeWeights=True, optimizer=optimizer,
                            max_relu_bound=max_relu_bound, adv_penalty=adv_penalty, unprotected=True, verbose=verbose)

        #set the vgg weights
        self.ourVGG.model.load_weights(vggModelLocation)

        #set protected flag
        self.unprotected = unprotected

        #instantiate our CAE with residual connections
        self.input_dimension, self.output_dimension = input_dimension, np.copy(output_dimension)

        self.loss_threshold, self.number_of_classes = np.copy(loss_threshold), np.copy(number_of_classes)
        self.dropoutRate, self.max_relu_bound = dropout_rate, np.copy(max_relu_bound)
        self.patience = np.copy(patience)
        # self.learning_rate, self.learning_rate_drop = np.copy(learning_rate), np.copy(learning_rate_drop)
        self.image_size = 32
        self.num_channels = 3
        self.num_labels = number_of_classes
        self.penaltyCoeff = adv_penalty
        
        if (verbose):
            print("input dimension: %s"%str(self.input_dimension))
    
        # define input layer
        self.inputLayer = layers.Input(shape=self.input_dimension)
        self.advInputLayer = layers.Input(shape=self.input_dimension)
        previousLayer = self.advInputLayer

        #following the architecture shown in figure 3 of the hlrgd paper, [ https://openaccess.thecvf.com/content_cvpr_2018/papers/Liao_Defense_Against_Adversarial_CVPR_2018_paper.pdf ]
        #
        self.hiddenEncoderLayers = dict()
        #first encoder block
        previousLayer = layers.Conv2D(numberOfFilters[0], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2D(numberOfFilters[0], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        if (self.dropoutRate > 0):
            previousLayer = Dropout(self.dropoutRate)(previousLayer)
        self.hiddenEncoderLayers[0] = previousLayer

        #second encoder block
        previousLayer = layers.Conv2D(numberOfFilters[1], kernel_size=(3,3), strides=(2, 2), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2D(numberOfFilters[1], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2D(numberOfFilters[1], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        if (self.dropoutRate > 0):
            previousLayer = Dropout(self.dropoutRate)(previousLayer)
        self.hiddenEncoderLayers[1] = previousLayer

        #third encoder block
        previousLayer = layers.Conv2D(numberOfFilters[2], kernel_size=(3,3), strides=(2, 2), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2D(numberOfFilters[2], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2D(numberOfFilters[2], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        if (self.dropoutRate > 0):
            previousLayer = Dropout(self.dropoutRate)(previousLayer)
        self.hiddenEncoderLayers[2] = previousLayer

        #fourth encoder block
        previousLayer = layers.Conv2D(numberOfFilters[3], kernel_size=(3,3), strides=(2, 2), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2D(numberOfFilters[3], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2D(numberOfFilters[3], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        if (self.dropoutRate > 0):
            previousLayer = Dropout(self.dropoutRate)(previousLayer)
        self.hiddenEncoderLayers[3] = previousLayer

        #fifth encoder block
        previousLayer = layers.Conv2D(numberOfFilters[4], kernel_size=(3,3), strides=(2, 2), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2D(numberOfFilters[4], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2D(numberOfFilters[4], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        if (self.dropoutRate > 0):
            previousLayer = Dropout(self.dropoutRate)(previousLayer)
        self.hiddenEncoderLayers[4] = previousLayer

        #first decoding block
        #fuse inputs (i.e. concatenate the residual connection with an upscaled version of the previousLayer
        previousLayer = layers.UpSampling2D((2, 2), interpolation='nearest')(previousLayer)
        previousLayer = layers.concatenate([previousLayer, self.hiddenEncoderLayers[3]])
        #perform convolutions
        previousLayer = layers.Conv2DTranspose(numberOfFilters[5], kernel_size=(3,3), strides=(2, 2), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2DTranspose(numberOfFilters[5], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2DTranspose(numberOfFilters[5], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        if (self.dropoutRate > 0):
            previousLayer = Dropout(self.dropoutRate)(previousLayer)

        #second decoding block
        #fuse inputs (i.e. concatenate the residual connection with an upscaled version of the previousLayer
        # previousLayer = layers.UpSampling2D((2, 2), interpolation='nearest')(previousLayer)
        previousLayer = layers.concatenate([previousLayer, self.hiddenEncoderLayers[2]])
        #perform convolutions
        previousLayer = layers.Conv2DTranspose(numberOfFilters[6], kernel_size=(3,3), strides=(2, 2), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2DTranspose(numberOfFilters[6], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2DTranspose(numberOfFilters[6], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        if (self.dropoutRate > 0):
            previousLayer = Dropout(self.dropoutRate)(previousLayer)

        #third decoding block
        #fuse inputs (i.e. concatenate the residual connection with an upscaled version of the previousLayer
        # previousLayer = layers.UpSampling2D((2, 2), interpolation='nearest')(previousLayer)
        previousLayer = layers.concatenate([previousLayer, self.hiddenEncoderLayers[1]])
        #perform convolutions
        previousLayer = layers.Conv2DTranspose(numberOfFilters[7], kernel_size=(3,3), strides=(2, 2), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2DTranspose(numberOfFilters[7], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2DTranspose(numberOfFilters[7], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        if (self.dropoutRate > 0):
            previousLayer = Dropout(self.dropoutRate)(previousLayer)

        #fourth decoding block
        #fuse inputs (i.e. concatenate the residual connection with an upscaled version of the previousLayer
        # previousLayer = layers.UpSampling2D((2, 2), interpolation='nearest')(previousLayer)
        previousLayer = layers.concatenate([previousLayer, self.hiddenEncoderLayers[0]])
        #perform convolutions
        previousLayer = layers.Conv2DTranspose(numberOfFilters[8], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousLayer = layers.Conv2DTranspose(numberOfFilters[8], kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(previousLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        if (self.dropoutRate > 0):
            previousLayer = Dropout(self.dropoutRate)(previousLayer)

        #1x1 convolutional layer
        previousLayer = layers.Conv2DTranspose(1, kernel_size=(1,1), strides=(1, 1), padding='same', activation='relu')(previousLayer)

        #compute the output
        #\hat{x} = x^{\asterisk} - d\hat{x}
        duOutput = self.inputLayer - previousLayer

        #compute vgg penultimate layer post-activation outputs when the vgg subnet is stimulated with duOutput and the benign sample
        # inputList = [self.inputLayer, self.inputLayer]
        # duList = [duOutput, duOutput]
        vggOutputBenign, vggPenultimate = self.ourVGG.model([self.inputLayer, self.inputLayer])
        vggOutputDU, vggPenultimateDU = self.ourVGG.model([duOutput, duOutput])


        #assign output layer as the evaluation of vgg that depends on the (potentially) adv. input
        self.outputLayer = vggOutputDU

        #formulate our loss function
        #define our custom loss function
        #this is L1 norm of the difference between the vggOutput
        def customLossWrapper(vggPenultimateDU, vggPenultimate, penaltyCoeff = self.penaltyCoeff):
            def customLoss(y_true, y_pred):
                return K.sum(K.abs(vggPenultimateDU - vggPenultimate))
            return customLoss
        # optimization details
        # def lr_scheduler(epoch):
        #     return self.learning_rate * (0.5 ** (epoch // self.learning_rate_drop))
        # self.lr_scheduler = lr_scheduler
        # self.reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
        # self.sgd = tf.keras.optimizers.Nadam()#Adadelta(learning_rate=self.learning_rate)
        # self.sgd = tf.keras.optimizers.SGD(lr=self.learning_rate, decay=0.000001, momentum=0.9, nesterov=True)
        if (optimizer==None):
            self.sgd = tf.keras.optimizers.Nadam()#Adadelta(learning_rate=self.learning_rate)
            self.reduceLR = None
        elif(optimizer=='SGD'):
            def lr_scheduler(epoch):
                return 0.1 * (0.5 ** (epoch // 20))#learning_rate * (0.5 ** (epoch // 20))

            self.reduceLR = keras.callbacks.LearningRateScheduler(lr_scheduler)
            self.sgd = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        #set up data augmentation
        self.generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                            samplewise_center=False,  # set each sample mean to 0
                                            featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                            samplewise_std_normalization=False,  # divide each input by its std
                                            zca_whitening=False,  # apply ZCA whitening
                                            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                                            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                            horizontal_flip=True,  # randomly flip images
                                            vertical_flip=False)

        # convert self.hiddenAdvLayers to a list for the model compilation, ascending order of keys is order of layers
        # outputsList is a list of outputs of the model constructed so that the first entry is the true output (ie prediction) layer
        # and each subsequent (i, i+1)th entries are the pair of hiddenAdvLayer, hiddenBenignLayer activations
        # this is going to be useful for calculating the MAE between benignly and adversarially induced hidden states
        # outputsList = [self.outputLayer]
        # for curHiddenLayer in range(len(self.ourVGG.hiddenAdvModelOutputs)):
        #     outputsList.append(self.ourVGG.hiddenAdvModelOutputs[curHiddenLayer])
        #     outputsList.append(self.ourVGG.hiddenModelOutputs[curHiddenLayer])

        # instantiate and compile the model
        self.customLossWrapper = customLossWrapper
        self.model = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=[self.outputLayer], name='hlrgd_vgg16')
        self.model.compile(loss=customLossWrapper(vggPenultimateDU, vggPenultimate, self.penaltyCoeff), metrics=['acc'], optimizer=self.sgd)

        # double check weight trainability bug
        allVars = self.model.variables
        trainableVars = self.model.trainable_variables
        allVarNames = [self.model.variables[i].name for i in range(len(self.model.variables))]
        trainableVarNames = [self.model.trainable_variables[i].name for i in range(len(self.model.trainable_variables))]
        nonTrainableVars = np.setdiff1d(allVarNames, trainableVarNames)

        if (verbose):
            if (len(nonTrainableVars) > 0):
                print('the following variables are set to non-trainable; ensure that this is correct before publishing!!!!')
            print(nonTrainableVars)
            self.model.summary()

        #set data statistics to default values
        self.mean = 0
        self.stddev = 1


    # handle data augmentation with multiple inputs (example found on https://stackoverflow.com/questions/49404993/keras-how-to-use-fit-generator-with-multiple-inputs
    #so thanks to loannis and Julian
    def multiInputDataGenerator(self, X1, X2, Y, batch_size):
        genX1 = self.generator.flow(X1, Y, batch_size=batch_size)
        genX2 = self.generator.flow(X2, Y, batch_size=batch_size)

        while True:
            X1g = genX1.next()
            X2g = genX2.next()
            yield [X1g[0], X2g[0]], X1g[1]

    #this method trains the HLRGD
    #inputData is a list of training matrices, the 0th is adversarial (i.e., to be denoised), the 2nd is the benign input
    def train(self, inputTrainingData, trainingTargets, inputValidationData, validationTargets, training_epochs=1, normed=False, monitor='val_loss',
              patience=defaultPatience, model_path=None, keras_batch_size=None, validation_split=0.1, adversarialOrder=0,
              dataAugmentation=False):
        
        #if a path isn't provided by caller, just use the current time for restoring best weights from fit
        if (model_path is None):
            model_path = os.path.join('/tmp/models/', 'hlrgd_vgg16_'+str(int(round(time.time()*1000))))
         
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
                                                horizontal_flip=False,  # randomly flip images
                                                vertical_flip=False)
            self.generator.fit(trainingData[0])
            history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, keras_batch_size),
                                     steps_per_epoch=trainingData[0].shape[0] // keras_batch_size,
                                     epochs=training_epochs, validation_data=(validationData, validationTargets),
                                     callbacks=[earlyStopper, checkpoint], verbose=1) #self.reduce_lr
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
                                     callbacks=[earlyStopper, checkpoint], verbose=1) #self.reduce_lr

        self.model.load_weights(model_path)
        loss, acc = history.history['loss'], history.history['val_acc']
        return loss, acc, model_path

    def evaluate(self, inputData, targets, batchSize=None):
        evalData = [self.normalize(inputData[0], storeStats=False), self.normalize(inputData[1], storeStats=False)]
        fullEval = self.model.evaluate(evalData, targets, batch_size=batchSize)
        return fullEval

    #method to read model from the disk
    def readModelFromDisk(self, pathToFile, vggModelLocation):
        #rebuild the model
        self.buildModel(self.input_dimension, self.output_dimension, vggModelLocation, self.number_of_classes,
                        loss_threshold=self.loss_threshold, patience=self.patience, dropout_rate=self.dropoutRate,
                        max_relu_bound=self.max_relu_bound, adv_penalty=self.penaltyCoeff, unprotected=self.unprotected,
                        verbose=False)
        # set the vgg weights
        self.model.load_weights(pathToFile)
        # #read in the picklebox
        pickleBox = pickle.load(open(pathToFile+'_pickle', 'rb'))
        # # self.bottleneckLayer = pickleBox['bottleneckLayer']
        # # self.hiddenEncodingLayer = pickleBox['hiddenEncodingLayer']
        # # self.inputLayer = pickleBox['inputLayer']
        self.mean, self.std = pickleBox['scaleMean'], pickleBox['scaleSTD']

        #first read in the inner model
        # self.ourVGG.readModelFromDisk('inner_'+pathToFile)

        # #read in the picklebox
        # pickleBox = pickle.load(open(pathToFile+'_pickle', 'rb'))
        # # self.bottleneckLayer = pickleBox['bottleneckLayer']
        # # self.hiddenEncodingLayer = pickleBox['hiddenEncodingLayer']
        # # self.inputLayer = pickleBox['inputLayer']
        # self.chosenActivation = pickleBox['chosenActivation']

        # formulate our loss function
        # define our custom loss function
        # this is L1 norm of the difference between the vggOutput
        # if (not self.unprotected):
        #     def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
        #         def customLoss(y_true, y_pred):
        #             return K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(0.00000001+tf.cast(tf.shape(benProjs)[0], tf.float32))
        #         return customLoss
        # else:#if we are using an unprotected model, don't force the  machine to calculate this too
        #     def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
        #         def customLoss(y_true, y_pred):
        #             return K.categorical_crossentropy(y_true, y_pred)
        #         return customLoss
        # self.customLossWrapper = customLossWrapper(self.benProjs, self.advProjs)
        # #load the model
        # self.model = load_model(pathToFile, custom_objects={'customLossWrapper': self.customLossWrapper})


    # this routine is used to collect statistics on training data, as well as to preprocess the training data by normalizing
    # i.e. centering and dividing by standard deviation
    def normalize(self, inputData, storeStats=False):
        if (storeStats):
            self.mean = np.mean(inputData)
            self.stddev = np.std(inputData)
        outputData = (inputData - self.mean) / (self.stddev + 0.0000001)
        return outputData

    # routine to get a pointer to the optimizer of this model
    def getOptimizer(self):
        if (self.ourVGG is not None):
            return self.sgd, self.ourVGG.getOptimizer()
        else:
            return self.sgd
########################################################################################################################

########################################################################################################################
if __name__ == "__main__":
    #define test parameters
    # define parameters
    verbose = True
    testFraction = 0.01
    numberOfClasses = 10
    numberOfAdvSamples = 100
    trainingEpochs = 1
    maxCarliniIts = 10000
    powers = np.arange(0, 2, 1.1)

    # set up data
    # input image dimensions
    img_rows, img_cols = 32, 32

    inputDimension = (img_rows, img_cols, 1)
    # read in data
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
    # x_train = (x_train - 0.13088295) / (0.30840662)
    # x_test = (x_test - 0.13088295) / (0.30840662)
    # x_train = (x_train - 0.28654176) / (0.35317212)
    # x_test = (x_test - 0.28654176) / (0.35317212)

    # #re-encode labels as one-hot
    # t_trainOneHot = tf.keras.utils.to_categorical(t_train, preNumberOfClasses)
    # t_testOneHot = tf.keras.utils.to_categorical(t_test, preNumberOfClasses)

    # flatten the cifar data
    # x_train = x_train.reshape([x_train.shape[0],(32*32*3)])
    # x_test = x_test.reshape([x_test.shape[0],(32*32*3)])

    # split cifarData by classes so we can distribute it
    task_permutation = []
    trainXs, trainTargets = [], []
    testXs, testTargets = [], []
    # split cifarData by classes so we can distribute it
    task_permutation = []
    trainXs, trainTargets = [], []
    testXs, testTargets = [], []

    for t in range(numberOfClasses):
        # arrange training and validation data
        curClassIndicesTraining = np.where(t_train[:] == t)
        curXTrain = x_train[curClassIndicesTraining, :, :][0]
        trainXs.append(curXTrain)
        trainTargets.append(t * np.ones([curXTrain.shape[0], 1]))

        # arrange testing data
        curClassIndicesTesting = np.where(t_test[:] == t)
        curXTest = x_test[curClassIndicesTesting, :, :][0]
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

    trainX, testX, trainY, testY = model_selection.train_test_split(stackedData[:256], stackedTargets[:256],
                                                                    test_size=0.25, random_state=42)
    output_dimension = numberOfClasses
    hlrVGGModel = HLVGG(inputDimension, output_dimension, number_of_classes=numberOfClasses, adv_penalty=0.00005,
                       loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=0,
                       max_relu_bound=1.1, verbose=False)
    vggWeightsList = hlrVGGModel.storeModelToDisk(os.path.join(os.getcwd(),'vggModel.h5'))
    ourModel = HLRGDVGG(inputDimension, output_dimension, vggModelLocation=os.path.join(os.getcwd(),'vggModel.h5'), number_of_classes=numberOfClasses, adv_penalty=0.00005,
                       loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=0.5,
                       max_relu_bound=1.1, verbose=False)

    ourModel.train([np.expand_dims(trainX, axis=3), np.expand_dims(trainX, axis=3)], tf.keras.utils.to_categorical(trainY, num_classes=numberOfClasses),
                    [np.expand_dims(trainX, axis=3), np.expand_dims(trainX, axis=3)], tf.keras.utils.to_categorical(trainY, num_classes=numberOfClasses), 
                    training_epochs=1, monitor='val_loss', patience=25, dataAugmentation=True,
                    model_path=os.path.join(os.getcwd(),'hlrgdvggtestmodel.h5'), keras_batch_size=128)
    ourModel.storeModelToDisk('hlrgdvggTestModel.h5')
    ourModel.readModelFromDisk('hlrgdvggTestModel.h5', vggModelLocation=os.path.join(os.getcwd(),'vggModel.h5'))
    print(ourModel.evaluate([np.expand_dims(testX, axis=3), np.expand_dims(testX, axis=3)], tf.keras.utils.to_categorical(testY, numberOfClasses)))
########################################################################################################################
