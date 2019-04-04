import os
from ROOT import *
import root_numpy
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, merge, LSTM, Input
from keras.layers.normalization import BatchNormalization
import matplotlib, json, glob, math
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from random import randint
seed = 7
np.random.seed(seed) #init for reproducibilty

##Input
maxtracks_read = 100 # max number of tracks to read
maxtracks_train = 25 # max number of tracks to use in the training 

filename = 'ntuHevjin.root'
file=TFile(filename, 'R')
tree=file.Get('PDsecondTree')

evtcuts = '((evtNumber % 10) >= 8)' # leave out 20% of events for testing

vInput=root_numpy.tree2array(tree, branches=['trkPt', 'trkEta', 'trkPhi', 'trkDxy', 'trkDz', 'trkIsInJet', 'trkIsHighPurity', 'trkCharge'], selection=evtcuts)
vInput=root_numpy.rec2array(vInput)

nspec = 2 # numebr of feature not used in the training
nfeat = len(vInput[0]) - nspec

vPt = vInput[:,0]
vEta = vInput[:,1]
vPhi = vInput[:,2]
vDxy = vInput[:,3]
vDz = vInput[:,4]
vtrkIsInJet = vInput[:,-3] 
vtrkIsHighPurity = vInput[:,-2]
vQ = vInput[:,-1]

##Shape formatting and zero padding
vInput = np.zeros([len(vPt), maxtracks_read, nfeat])

for i in range(len(vPt)):
    for j in range(len(vPt[i])):
        if j >= maxtracks_read:
            break
        ##Tracks selection
        if vtrkIsInJet[i][j] == 0 or vtrkIsHighPurity[i][j] == 0 :
            continue
        if abs(vDz[i][j]) > 1.0:
            continue
        if math.isnan(vDxy[i][j]):
            continue
        vInput[i][j][0] = vPt[i][j]
        vInput[i][j][1] = vEta[i][j]
        vInput[i][j][2] = vPhi[i][j]
        vInput[i][j][3] = vDxy[i][j]
        vInput[i][j][4] = vDz[i][j]
        vInput[i][j][-1] = vQ[i][j]

vInput.view('f8,f8,f8,f8,f8,f8').sort(order=['f0'], axis=1) #Ordering by increasing Pt (Leonardo's magic)

vInput = vInput[:,-maxtracks_train:,:] #Only using the most energetic maxtracks_train tracks to train

##Labels
vLabel=root_numpy.tree2array(tree, branches=['ssbLund'], selection=evtcuts)
vLabel=root_numpy.rec2array(vLabel)
vLabel[vLabel == 531] = 1
vLabel[vLabel == -531] = 0

##Weights
vWeights=root_numpy.tree2array(tree, branches=['evtWeight'], selection=evtcuts)
vWeights=root_numpy.rec2array(vWeights)
vWeights=vWeights.reshape((vWeights.shape[0],))

print '---- input loaded -----'

##Reading the model
model = load_model('testLSTM.h5')

print '---- model loaded ----'

score = model.predict(vInput)
score = score.reshape(len(score))
pred = np.copy(score)
pred[pred >= 0.5 ] = 1
pred[pred < 0.5 ] = 0
nevt = len(pred)
sumweight = 0
for i in vWeights:
    sumweight = sumweight + i

rt = 0
wt = 0
for i in range(nevt):
    if pred[i] == vLabel[i]:
        rt =  rt + vWeights[i]
    else:
        wt = wt + vWeights[i]

avgW = float(wt)/(float(rt+wt))
avgWerr = (TEfficiency.AgrestiCoull(float(rt+wt), float(wt), 0.687, 1)-TEfficiency.AgrestiCoull(float(rt+wt), float(wt), 0.687, 0) )/2

totD2 = 0.
for i in range(nevt):
    if score[i] >= 0.5:
        w = 1 - score[i]
    else:
        w = score[i]
    totD2 += vWeights[i]*((1 - 2*w)**2 ) / sumweight

test_loss, test_acc = model.evaluate(vInput, vLabel)
print('avgw from acc (bias not adressed: %.2f%%' % ((1-test_acc)*100))
print('avgw from count: (%.2f +- %.2f)%%' % (avgW*100 , avgWerr*100))
print('avgD2 from count: %.4f' % ((1 - 2*avgW)**2))
print('avgD2 from per-event: %.4f' % (totD2))
