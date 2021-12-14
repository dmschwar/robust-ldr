########################################################################################################################
#
# Author: David Schwartz, June, 9, 2020
#
# This script plots the aggregated results of the cross validation experiment in fine_tuning_experiment.py
########################################################################################################################

########################################################################################################################
#imports
import os, sys, copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
########################################################################################################################


########################################################################################################################
cvResultsFileName = 'crossValResultscifar10'
pickleFile = pickle.load(open(os.path.join(os.getcwd(),cvResultsFileName+'.pickle'), 'rb'))
crossValResults = copy.copy(pickleFile)
print("keys")
print(list(crossValResults.keys()))
print(list(crossValResults['averageLayerwiseUndefMAEs'].keys()))
print(crossValResults['averageLayerwiseUndefMAEs'][0])
print("data loaded")
print(list(crossValResults.keys()))
noisyTraining = crossValResults['noisyTraining']
powersAcc = crossValResults['budgets'] #uncomment after the next round of resumes is done
noisePowers = crossValResults['noiseBudgets']
########################################################################################################################

########################################################################################################################
#load variables
numberOfFolds = len(list(crossValResults['Acc'].keys()))
numberOfNoiseMagnitudes = len(crossValResults['Acc'][0]['undefended'])
numberOfAGNMagnitudes = len(crossValResults['noiseAcc'][0]['undefended'])
maeCVKey = list(crossValResults['averageLayerwiseUndefMAEs'].keys())[0]
#this is no longer a dimension that matters: maeNoiseKey = list(crossValResults['averageLayerwiseUndefMAEs'][maeCVKey].keys())[0]
numberOfLayers = len(crossValResults['averageLayerwiseUndefMAEs'][maeCVKey])
aggregatedResultsMats = dict()

#collect aggregated results across CV experiment
aggregatedAccsUndef, aggregatedAccsFIM, aggregatedAccsHLDR, aggregatedAccsHLRGD, aggregatedAccsAGN = -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes)),\
                                                                                                      -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes)),\
                                                                                                      -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes)),\
                                                                                                      -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes)),\
                                                                                                      -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes))
aggregatedMAEsUndef, aggregatedMAEsFIM, aggregatedMAEsHLDR, aggregatedMAEsHLRGD, aggregatedMAEsAGN = -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                      -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                      -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                      -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                      -1*np.ones(shape=(numberOfFolds, numberOfLayers))
aggregatedAGNAccsUndef, aggregatedAGNAccsFIM, aggregatedAGNAccsHLDR, aggregatedAGNAccsHLRGD, aggregatedAGNAccsAGN = -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes)), \
                                                                                                                          -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes)),\
                                                                                                                          -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes)),\
                                                                                                                          -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes)), \
                                                                                                                          -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes))
aggregatedAGNMAEsUndef, aggregatedAGNMAEsFIM, aggregatedAGNMAEsHLDR, aggregatedAGNMAEsHLRGD, aggregatedAGNMAEsAGN = -1*np.ones(shape=(numberOfFolds, numberOfLayers)), \
                                                                                                                          -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                                          -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                                          -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                                          -1*np.ones(shape=(numberOfFolds, numberOfLayers))
for cvIndex in range(numberOfFolds):
    #aggregate adversarial results
    aggregatedAccsUndef[cvIndex, :] = crossValResults['Acc'][cvIndex]['undefended']
    aggregatedAccsFIM[cvIndex, :] = crossValResults['Acc'][cvIndex]['fim']
    aggregatedAccsHLDR[cvIndex, :] = crossValResults['Acc'][cvIndex]['hldr']
    aggregatedAccsHLRGD[cvIndex, :] = crossValResults['Acc'][cvIndex]['hlrgd']
    aggregatedAccsAGN[cvIndex, :] = crossValResults['Acc'][cvIndex]['agn']
    aggregatedMAEsUndef[cvIndex, :] = np.array(crossValResults['averageLayerwiseUndefMAEs'][cvIndex])
    aggregatedMAEsFIM[cvIndex, :] = np.array(crossValResults['averageLayerwiseFIMMAEs'][cvIndex])
    aggregatedMAEsHLDR[cvIndex, :] = np.array(crossValResults['averageLayerwiseHLDRMAEs'][cvIndex])
    aggregatedMAEsHLRGD[cvIndex, :] = np.array(crossValResults['averageLayerwiseHLRGDMAEs'][cvIndex])
    if (noisyTraining):
        aggregatedMAEsAGN[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNMAEs'][cvIndex])
    #aggregate AGN results
    aggregatedAGNAccsUndef[cvIndex, :] = crossValResults['noiseAcc'][cvIndex]['undefended']
    aggregatedAGNAccsFIM[cvIndex, :] = crossValResults['noiseAcc'][cvIndex]['fim']
    aggregatedAGNAccsHLDR[cvIndex, :] = crossValResults['noiseAcc'][cvIndex]['hldr']
    aggregatedAGNAccsHLRGD[cvIndex, :] = crossValResults['noiseAcc'][cvIndex]['hlrgd']
    aggregatedAGNAccsAGN[cvIndex, :] = crossValResults['noiseAcc'][cvIndex]['agn']
    aggregatedAGNMAEsUndef[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNUndefMAEs'][cvIndex])
    aggregatedAGNMAEsFIM[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNFIMMAEs'][cvIndex])
    aggregatedAGNMAEsHLDR[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNHLDRMAEs'][cvIndex])
    aggregatedAGNMAEsHLRGD[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNHLRGDMAEs'][cvIndex])
    if (noisyTraining):
        aggregatedAGNMAEsAGN[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNAGNMAEs'][cvIndex])


#calculate aggregated statistics of interest (i.e. mean across the cvIndex axis
meanAdvAccs, stdAdvAccs = dict(), dict()
meanNoiseAccs, stdNoiseAccs = dict(), dict()

meanAdvAccs['undefended'] = np.mean(aggregatedAccsUndef, axis=0)
meanAdvAccs['fim'] = np.mean(aggregatedAccsFIM, axis=0)
meanAdvAccs['hldr'] = np.mean(aggregatedAccsHLDR, axis=0)
meanAdvAccs['hlrgd'] = np.mean(aggregatedAccsHLRGD, axis=0)
if (noisyTraining):
    meanAdvAccs['agn'] = np.mean(aggregatedAccsAGN, axis=0)

meanNoiseAccs['undefended'] = np.mean(aggregatedAGNAccsUndef, axis=0)
meanNoiseAccs['fim'] = np.mean(aggregatedAGNAccsFIM, axis=0)
meanNoiseAccs['hldr'] = np.mean(aggregatedAGNAccsHLDR, axis=0)
meanNoiseAccs['hlrgd'] = np.mean(aggregatedAGNAccsHLRGD, axis=0)
if (noisyTraining):
    meanNoiseAccs['agn'] = np.mean(aggregatedAGNAccsAGN, axis=0)

stdAdvAccs['undefended'] = 1.96*np.std(aggregatedAccsUndef, axis=0)/np.sqrt(numberOfFolds)
stdAdvAccs['fim'] = 1.96*np.std(aggregatedAccsFIM, axis=0)/np.sqrt(numberOfFolds)
stdAdvAccs['hldr'] = 1.96*np.std(aggregatedAccsHLDR, axis=0)/np.sqrt(numberOfFolds)
stdAdvAccs['hlrgd'] = 1.96*np.std(aggregatedAccsHLRGD, axis=0)/np.sqrt(numberOfFolds)
if (noisyTraining):
    stdAdvAccs['agn'] = 1.96*np.std(aggregatedAccsAGN, axis=0)/np.sqrt(numberOfFolds)

stdNoiseAccs['undefended'] = 1.96*np.std(aggregatedAGNAccsUndef, axis=0)/np.sqrt(numberOfFolds)
stdNoiseAccs['fim'] = 1.96*np.std(aggregatedAGNAccsFIM, axis=0)/np.sqrt(numberOfFolds)
stdNoiseAccs['hldr'] = 1.96*np.std(aggregatedAGNAccsHLDR, axis=0)/np.sqrt(numberOfFolds)
stdNoiseAccs['hlrgd'] = 1.96*np.std(aggregatedAGNAccsHLRGD, axis=0)/np.sqrt(numberOfFolds)
if (noisyTraining):
    stdNoiseAccs['agn'] = 1.96*np.std(aggregatedAGNAccsAGN, axis=0)/np.sqrt(numberOfFolds)

meanAverageLayerwiseUndefMAEs = np.mean(aggregatedMAEsUndef, axis=0)
meanAverageLayerwiseHLDRMAEs = np.mean(aggregatedMAEsHLDR, axis=0)
meanAverageLayerwiseHLRGDMAEs = np.mean(aggregatedMAEsHLRGD, axis=0)
meanAverageLayerwiseFIMMAEs = np.mean(aggregatedMAEsFIM, axis=0)
if (noisyTraining):
    meanAverageLayerwiseAGNMAEs = np.mean(aggregatedMAEsAGN, axis=0)

meanAverageLayerwiseAGNUndefMAEs = np.mean(aggregatedAGNMAEsUndef, axis=0)
meanAverageLayerwiseAGNHLDRMAEs = np.mean(aggregatedAGNMAEsHLDR, axis=0)
meanAverageLayerwiseAGNHLRGDMAEs = np.mean(aggregatedAGNMAEsHLRGD, axis=0)
meanAverageLayerwiseAGNFIMMAEs = np.mean(aggregatedAGNMAEsFIM, axis=0)
if (noisyTraining):
    meanAverageLayerwiseAGNAGNMAEs = np.mean(aggregatedAGNMAEsAGN, axis=0)

stdAverageLayerwiseUndefMAEs = 1.96*np.std(aggregatedMAEsUndef, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseHLDRMAEs = 1.96*np.std(aggregatedMAEsHLDR, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseHLRGDMAEs = 1.96*np.std(aggregatedMAEsHLRGD, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseFIMMAEs = 1.96*np.std(aggregatedMAEsFIM, axis=0)/np.sqrt(numberOfFolds)
if (noisyTraining):
    stdAverageLayerwiseAGNMAEs = 1.96*np.std(aggregatedMAEsAGN, axis=0)/np.sqrt(numberOfFolds)

stdAverageLayerwiseAGNUndefMAEs = 1.96*np.std(aggregatedAGNMAEsUndef, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseAGNHLDRMAEs = 1.96*np.std(aggregatedAGNMAEsHLDR, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseAGNHLRGDMAEs = 1.96*np.std(aggregatedAGNMAEsHLRGD, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseAGNFIMMAEs = 1.96*np.std(aggregatedAGNMAEsFIM, axis=0)/np.sqrt(numberOfFolds)
if (noisyTraining):
    stdAverageLayerwiseAGNAGNMAEs = 1.96*np.std(aggregatedAGNMAEsAGN, axis=0)/np.sqrt(numberOfFolds)
########################################################################################################################

########################################################################################################################
#plot stuff
fig = plt.figure()
ax = fig.add_subplot(111)
# plt.title('adversarial accuracy')
plt.ylabel('accuracy')
plt.xlabel('adversarial budget ($\epsilon$)')
powersAccPlot = copy.copy(powersAcc)
powersAccPlot.insert(0, 0)
powersAccAx = np.array(powersAccPlot)*(255)
plt.errorbar(powersAccAx, meanAdvAccs['hldr'], stdAdvAccs['hldr'], label='HLDR', marker='d')
plt.errorbar(powersAccAx, meanAdvAccs['hlrgd'], stdAdvAccs['hlrgd'], label='HGD', marker='x')
plt.errorbar(powersAccAx, meanAdvAccs['fim'], stdAdvAccs['fim'], label='FIMR', marker='*')
if (noisyTraining):
    plt.errorbar(powersAccAx, meanAdvAccs['agn'], stdAdvAccs['agn'], label='AGNT', linestyle='-.')
plt.legend()
plt.savefig('aggregatedACCVsBudget_no_labels.pdf')
texts = []
for i in range(len(meanAdvAccs['undefended'])):
    texts.append(plt.text(powersAccAx[i], meanAdvAccs['undefended'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(meanAdvAccs['undefended'][i]))))
for i in range(len(meanAdvAccs['hldr'])):
    texts.append(plt.text(powersAccAx[i], meanAdvAccs['hldr'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(meanAdvAccs['hldr'][i]))))
for i in range(len(meanAdvAccs['hlrgd'])):
    texts.append(plt.text(powersAccAx[i], meanAdvAccs['hlrgd'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(meanAdvAccs['hlrgd'][i]))))
for i in range(len(meanAdvAccs['fim'])):
    texts.append(plt.text(powersAccAx[i], meanAdvAccs['fim'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(meanAdvAccs['fim'][i]))))
if (noisyTraining):
    for i in range(len(meanAdvAccs['agn'])):
        texts.append(plt.text(powersAccAx[i], meanAdvAccs['agn'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(meanAdvAccs['agn'][i]))))
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
plt.savefig('aggregatedACCVsBudget_with_labels.pdf')
plt.show()


#marginal improvement plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylabel('accuracy gain')
plt.xlabel('adversarial budget ($\epsilon$)')
plt.errorbar(np.array(powersAccPlot)*(255), meanAdvAccs['hldr']-meanAdvAccs['undefended'], stdAdvAccs['hldr'], label='HLDR', marker='d')
plt.errorbar(np.array(powersAccPlot)*(255), meanAdvAccs['hlrgd']-meanAdvAccs['undefended'], stdAdvAccs['hlrgd'], label='HGD', marker='x')
plt.errorbar(np.array(powersAccPlot) * (255), meanAdvAccs['fim'] - meanAdvAccs['undefended'], stdAdvAccs['fim'], label='FIMR', marker='*')
if (noisyTraining):
    plt.errorbar(np.array(powersAccPlot) * (255), meanAdvAccs['agn'] - meanAdvAccs['undefended'], stdAdvAccs['agn'], label='AGNT', linestyle='-.')
plt.legend()
plt.savefig('aggregated_marg_improv_vs_budget_no_labels.pdf')
texts = []
# for i in range(len(advAccuracy['undefended'])):
#     texts.append(plt.text(powersAccAx[i], advAccuracy['undefended'][i], '(%s, %s)'%(str(powersAcc[i]), str(advAccuracy['undefended'][i]))))
for i in range(len(meanAdvAccs['hldr'])):
    texts.append(plt.text(powersAccAx[i], meanAdvAccs['hldr'][i]-meanAdvAccs['undefended'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(meanAdvAccs['hldr'][i]))))
for i in range(len(meanAdvAccs['hlrgd'])):
    texts.append(plt.text(powersAccAx[i], meanAdvAccs['hlrgd'][i]-meanAdvAccs['undefended'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(meanAdvAccs['hlrgd'][i]))))
for i in range(len(meanAdvAccs['fim'])):
    texts.append(plt.text(powersAccAx[i], meanAdvAccs['fim'][i]-meanAdvAccs['undefended'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(meanAdvAccs['fim'][i]))))
if (noisyTraining):
    for i in range(len(meanAdvAccs['agn'])):
        texts.append(plt.text(powersAccAx[i], meanAdvAccs['agn'][i]-meanAdvAccs['undefended'][i], '(%s, %s)'%(str(powersAccPlot[i]), str(meanAdvAccs['agn'][i]))))
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
plt.savefig('aggregated_marg_improv_vs_budget_with_labels.pdf')
plt.show()

#noise plots
#plot noise acc results
fig = plt.figure()
ax = fig.add_subplot(111)
# plt.title('AGN accuracy')
plt.ylabel('accuracy')
plt.xlabel('noise power ($\sigma$)')
noisePowersPlot = copy.copy(noisePowers)
noisePowersPlot.insert(0, 0)
plt.errorbar(np.array(noisePowersPlot)*(255), meanNoiseAccs['hldr'], stdNoiseAccs['hldr'], label='HLDR', marker='d')
plt.errorbar(np.array(noisePowersPlot)*(255), meanNoiseAccs['hlrgd'], stdNoiseAccs['hlrgd'], label='HGD', marker='x')
plt.errorbar(np.array(noisePowersPlot)*(255), meanNoiseAccs['fim'], stdNoiseAccs['fim'], label='FIMR', marker='*')
if (noisyTraining):
    plt.errorbar(np.array(noisePowersPlot)*(255), meanNoiseAccs['agn'], stdNoiseAccs['agn'], label='AGNT', linestyle='-.')
plt.legend()
plt.savefig('aggregatedACCVsBudget_no_labelsAGN.pdf')
texts = []
for i in range(len(meanNoiseAccs['undefended'])):
    texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['undefended'][i]))))
for i in range(len(meanNoiseAccs['hldr'])):
    texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['hldr'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['hldr'][i]))))
for i in range(len(meanNoiseAccs['hlrgd'])):
    texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['hlrgd'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['hlrgd'][i]))))
for i in range(len(meanNoiseAccs['fim'])):
    texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['fim'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['fim'][i]))))
if (noisyTraining):
    for i in range(len(meanNoiseAccs['agn'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['agn'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['agn'][i]))))
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
plt.savefig('aggregatedACCVsBudget_with_labelsAGN.pdf')
plt.show()


#marginal improvement plot
fig = plt.figure()
ax = fig.add_subplot(111)
# plt.title('adversarial accuracy')
plt.ylabel('accuracy gain')
plt.xlabel('noise power ($\sigma$)')
plt.errorbar(np.array(noisePowersPlot)*(255), meanNoiseAccs['hldr']-meanNoiseAccs['undefended'], stdNoiseAccs['hldr'], label='HLDR', linestyle='-.')
plt.errorbar(np.array(noisePowersPlot)*(255), meanNoiseAccs['hlrgd']-meanNoiseAccs['undefended'], stdNoiseAccs['hlrgd'], label='HGD', marker='x')
plt.errorbar(np.array(noisePowersPlot) * (255), meanNoiseAccs['fim'] - meanNoiseAccs['undefended'], stdNoiseAccs['fim'], label='FIMR', marker='*')
if (noisyTraining):
    plt.errorbar(np.array(noisePowersPlot) * (255), meanNoiseAccs['agn'] - meanNoiseAccs['undefended'], stdNoiseAccs['agn'], label='AGNT', linestyle='-.')
plt.legend()
plt.savefig('aggregated_marg_improv_vs_budget_no_labelsAGN.pdf')
texts = []
# for i in range(len(advAccuracy['undefended'])):
#     texts.append(plt.text(powersAccAx[i], advAccuracy['undefended'][i], '(%s, %s)'%(str(powersAcc[i]), str(advAccuracy['undefended'][i]))))
for i in range(len(meanNoiseAccs['hldr'])):
    texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['hldr'][i]-meanNoiseAccs['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['hldr'][i]))))
for i in range(len(meanNoiseAccs['hlrgd'])):
    texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['hlrgd'][i]-meanNoiseAccs['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['hlrgd'][i]))))
for i in range(len(meanNoiseAccs['fim'])):
    texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['fim'][i]-meanNoiseAccs['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['fim'][i]))))
if (noisyTraining):
    for i in range(len(meanNoiseAccs['agn'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['agn'][i]-meanNoiseAccs['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['agn'][i]))))
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
plt.savefig('aggregated_marg_improv_vs_budget_with_labelsAGN.pdf')
plt.show()

#perturbational error amplification plotting
#plot the maes
# for curLevelIndex in range(len(powersAcc)):
plt.figure()
plt.errorbar(range(numberOfLayers), meanAverageLayerwiseUndefMAEs, stdAverageLayerwiseUndefMAEs, marker='o', label='undefended')
plt.errorbar(range(numberOfLayers), meanAverageLayerwiseFIMMAEs, stdAverageLayerwiseFIMMAEs, marker='*', label='FIMR')
plt.errorbar(range(numberOfLayers), meanAverageLayerwiseHLRGDMAEs, stdAverageLayerwiseHLRGDMAEs, marker='x', label='HGD')
plt.errorbar(range(numberOfLayers), meanAverageLayerwiseHLDRMAEs, stdAverageLayerwiseHLDRMAEs, marker='d', label='HLDR')
if (noisyTraining):
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNMAEs, stdAverageLayerwiseAGNMAEs, marker='d', linestyle='-.', label='AGNT')

plt.ylabel(r'layer-wise perturbation: mean of ($\frac{\vert \mathbf{s}_i(\mathbf{x})-\mathbf{s}_i(\mathbf{x}_a)\vert}{\vert \mathbf{s}_i(\mathbf{x}) \vert} $)')
plt.xlabel('hidden layer index ($i$)')
plt.legend()
plt.savefig('layerwisePerturbationErrors_.pdf')
plt.show()

#plot noise maes
plt.figure()
plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNUndefMAEs, stdAverageLayerwiseAGNUndefMAEs, marker='o', label='undefended')
plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNFIMMAEs, stdAverageLayerwiseAGNFIMMAEs, marker='*', label='FIMR')
plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNHLRGDMAEs, stdAverageLayerwiseAGNHLRGDMAEs, marker='x', label='HGD')
plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNHLDRMAEs, stdAverageLayerwiseAGNHLDRMAEs, marker='d', label='HLDR')
if (noisyTraining):
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNAGNMAEs, stdAverageLayerwiseAGNAGNMAEs, marker='d', linestyle='-.', label='AGNT')

plt.ylabel(r'layer-wise perturbation: mean of ($\frac{\vert \mathbf{s}_i(\mathbf{x})-\mathbf{s}_i(\mathbf{x}_a)\vert}{\vert \mathbf{s}_i(\mathbf{x}) \vert} $)')
plt.xlabel('hidden layer index ($i$)')
plt.legend()
plt.savefig('layerwisePerturbationErrors_AGN.pdf')
plt.show()
########################################################################################################################
