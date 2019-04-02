import os
from ROOT import *
import root_numpy
import keras
from keras.models import Sequential, Model
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
file=TFile(filename, 'r')
tree=file.Get('PDsecondTree')
cuts = 'trkIsInJet==1 && trkIsHighPurity==1'

vInput=root_numpy.tree2array(tree, branches=['trkPt', 'trkEta', 'trkPhi','trkCharge'], selection=cuts)
vInput=root_numpy.rec2array(vInput)

nfeat = len(vInput[0])

vPt = vInput[:,0]
vEta = vInput[:,1]
vPhi = vInput[:,2]
vQ = vInput[:,-1]

##Shape formatting and zero padding
vInput = np.zeros([len(vPt), maxtracks_read, nfeat])

for i in range(len(vPt)):
    for j in range(len(vPt[i])):
        if j >= maxtracks_read:
            break
        vInput[i][j][0] = vPt[i][j]
        vInput[i][j][1] = vEta[i][j]
        vInput[i][j][2] = vPhi[i][j]
        vInput[i][j][-1] = vQ[i][j]

vInput.view('f8,f8,f8,f8').sort(order=['f0'], axis=1) #Ordering by Pt

vInput = vInput[:,-maxtracks_train:,:] #Only using the most energetic maxtracks_train tracks to train

vLabel=root_numpy.tree2array(tree, branches=['ssbLund'], selection='')
vLabel=root_numpy.rec2array(vLabel)
vLabel[vLabel == 531] = 1
vLabel[vLabel == -531] = 0

##Model Definition 
dropoutRate = 0.1
lstmOutDim = 20

Inputs = Input(shape=(maxtracks_train,nfeat)) # maxtracks_train, nfeat features

x = LSTM(lstmOutDim)
x = x(Inputs)
x = Dense(lstmOutDim, activation='relu',kernel_initializer='lecun_uniform')(x)
if dropoutRate != 0 :
    x = Dropout(dropoutRate)(x)
x = Dense(20, activation='relu',kernel_initializer='lecun_uniform')(x)
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
    , patience=3
    , min_lr=0.001
    )

history = model.fit(vInput, vLabel
    , batch_size=128
    , epochs=3
    , verbose=1
    , callbacks=[reduce_lr]
    , validation_split=0.2
    )

model.save('testLSTM.h5')


scores = model.evaluate(vInput, vLabel)
print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
print('w: %.2f%%' % ((1-scores[1])*100))

# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('png/'+'plot_history.png')
plt.clf()

