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
maxtracks = 50
filename = 'ntuHevjin.root'
file=TFile(filename, 'r')
tree=file.Get('PDsecondTree')
cuts = 'trkIsInJet==1 && trkIsHighPurity==1'

vInput=root_numpy.tree2array(tree, branches=['trkPt', 'trkEta', 'trkPhi','trkCharge'], selection=cuts)
vLabel=root_numpy.tree2array(tree, branches=['ssbLund'], selection='')
vInput=root_numpy.rec2array(vInput)

nfeat = len(vInput[0])

vPt = vInput[:,0]
vEta = vInput[:,1]
vPhi = vInput[:,2]
vQ = vInput[:,-1]

vInput = np.zeros([len(vPt), maxtracks, nfeat])

for i in range(len(vPt)):
    for j in range(len(vPt[i])):
        if j >= maxtracks:
            continue
        vInput[i][j][0] = vPt[i][j]
        vInput[i][j][1] = vEta[i][j]
        vInput[i][j][2] = vPhi[i][j]
        vInput[i][j][-1] = vQ[i][j]

vInput = vInput[:,::-1,:]

vLabel=root_numpy.rec2array(vLabel)
vLabel[vLabel == 531] = 1
vLabel[vLabel == -531] = 0

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

print str(rt) + ' ' + str(wt)

print('RT: %i%' % (rt))
print('WT: %i%' % (wt))
print('w: %.2f%%' % (( wt/(wt+rt) )*100))
