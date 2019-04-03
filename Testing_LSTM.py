import os
from ROOT import *
import root_numpy
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, merge, LSTM, Input
from keras.layers.normalization import BatchNormalization
import matplotlib, json, glob
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from random import randint
seed = 7
np.random.seed(seed) #init for reproducibilty

##Input
maxtracks_read = 100 # max number of tracks to read
maxtracks_train = 20 # max number of tracks to use in the training 

filename = 'ntuHevjin.root'
file=TFile(filename, 'R')
tree=file.Get('PDsecondTree')

evtcuts = '((evtNumber % 10) < 8)'

vInput=root_numpy.tree2array(tree, branches=['trkPt', 'trkEta', 'trkPhi', 'trkIsInJet', 'trkIsHighPurity', 'trkCharge'], selection=evtcuts)
vInput=root_numpy.rec2array(vInput)

nspec = 2
nfeat = len(vInput[0]) - nspec

vPt = vInput[:,0]
vEta = vInput[:,1]
vPhi = vInput[:,2]
vtrkIsInJet = vInput[:,3]
vtrkIsHighPurity = vInput[:,4]
vQ = vInput[:,-1]

##Shape formatting and zero padding
vInput = np.zeros([len(vPt), maxtracks_read, nfeat])

for i in range(len(vPt)):
    for j in range(len(vPt[i])):
        if j >= maxtracks_read:
            break
        if vtrkIsInJet == 0 or vtrkIsHighPurity == 0 :
            continue
        vInput[i][j][0] = vPt[i][j]
        vInput[i][j][1] = vEta[i][j]
        vInput[i][j][2] = vPhi[i][j]
        vInput[i][j][-1] = vQ[i][j]

vInput.view('f8,f8,f8,f8').sort(order=['f0'], axis=1) #Ordering by Pt

vInput = vInput[:,-maxtracks_train:,:] #Only using the most energetic maxtracks_train tracks to train

vLabel=root_numpy.tree2array(tree, branches=['ssbLund'], selection=evtcuts)
vLabel=root_numpy.rec2array(vLabel)
vLabel[vLabel == 531] = 1
vLabel[vLabel == -531] = 0

vWeights=root_numpy.tree2array(tree, branches=['evtWeight'], selection=evtcuts)
vWeights=root_numpy.rec2array(vWeights)
vWeights=vWeights.reshape((vWeights.shape[0],))

print '---- input loaded -----'

##Reading the model
model = load_model('testLSTM.h5')

print '---- model loaded ----'

pred = model.predict(vInput)
pred[pred>=0.5] = 1
pred[pred<0.5] = 0

rt = 0.
wt = 0.
for i in range(len(pred)):
    if pred[i] == vLabel[i]:
        rt = rt + 1
    else:
        wt = wt + 1

test_loss, test_acc = model.evaluate(vInput, vLabel)
print('w from acc: %.2f%%' % ((1-test_acc)*100))
