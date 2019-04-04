import os
from ROOT import *
import root_numpy
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, merge, LSTM, Input
from keras.layers.normalization import BatchNormalization
import matplotlib, json, glob, math
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from random import randint
seed = 10
np.random.seed(seed) #init for reproducibilty

##Input
maxtracks_read = 100 # max number of tracks to read
maxtracks_train = 25 # max number of tracks to use in the training 

filename = 'ntuHevjin.root'
file=TFile(filename, 'R')
tree=file.Get('PDsecondTree')

evtcuts = '((evtNumber % 10) < 8)' # leave out 20% of events for testing

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
        if math.isnan(vDxy[i][j]) :
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

##Model Definition 
dropoutRate = 0.5
lstmOutDim = 50

Inputs = Input(shape=(maxtracks_train,nfeat)) # maxtracks_train, nfeat features

x = LSTM(lstmOutDim)
x = x(Inputs)
x = Dense(lstmOutDim, activation='relu',kernel_initializer='lecun_uniform')(x)
if dropoutRate != 0 :
    x = Dropout(dropoutRate)(x)

x = Dense(50, activation='relu',kernel_initializer='lecun_uniform')(x)
if dropoutRate != 0 :
    x = Dropout(dropoutRate)(x)

x = Dense(25, activation='relu',kernel_initializer='lecun_uniform')(x)
if dropoutRate != 0 :
    x = Dropout(dropoutRate)(x)

x = Dense(10, activation='relu',kernel_initializer='lecun_uniform')(x)
if dropoutRate != 0 :
    x = Dropout(dropoutRate)(x)

predictions = Dense(1, kernel_initializer='lecun_uniform', activation='sigmoid')(x)

model = Model(inputs=Inputs, outputs=predictions)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

##Training model
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss'
    , factor=0.2
    , patience=5
    , min_lr=0.0001
    )

history = model.fit(vInput, vLabel
    , sample_weight = vWeights
    , batch_size=256
    , epochs=50
    , verbose=1
    , callbacks=[reduce_lr]
    , validation_split=0.2
    )

model.save('testLSTM.h5')

scores = model.evaluate(vInput, vLabel)
print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
w = 1-scores[1]
print('w: %.2f%%' % (w*100))
print('D^2: %.2f%%' % ( (1-2*w)*(1-2*w)*100 ))

# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('png/'+'plot_loss_history.png')
plt.clf()

# summarize history for acc
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('png/'+'plot_acc_history.png')
plt.clf()

