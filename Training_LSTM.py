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
maxtracks = 15
filename = 'ntuHevjin.root'
file=TFile(filename, 'r')
tree=file.Get('PDsecondTree')
cuts = 'trkIsInJet==1 && trkIsHighPurity==1'

vInput=root_numpy.tree2array(tree, branches=['trkPt', 'trkCharge'], selection=cuts)
vLabel=root_numpy.tree2array(tree, branches=['ssbLund'], selection='')
vInput=root_numpy.rec2array(vInput)

vPt = vInput[:,0]
vQ = vInput[:,1]

vInput = np.zeros([len(vPt), maxtracks, 2])

for i in range(len(vPt)):
    for j in range(len(vPt[i])):
        if j >= maxtracks:
            continue
        vInput[i][j][0] = vPt[i][j]
        vInput[i][j][1] = vQ[i][j]


vLabel=root_numpy.rec2array(vLabel)
vLabel[vLabel == 531] = 1
vLabel[vLabel == -531] = 0

##Model Definition 
dropoutRate = 0.1
lstmOutDim = 10

Inputs = Input(shape=(maxtracks,2)) # maxtracks, 2 features

x = LSTM(lstmOutDim)
x = x(Inputs)
x = Dense(lstmOutDim, activation='relu',kernel_initializer='lecun_uniform')(x)
if dropoutRate != 0 :
    x = Dropout(dropoutRate)(x)
x = Dense(5, activation='relu',kernel_initializer='lecun_uniform')(x)

predictions = Dense(1, kernel_initializer='lecun_uniform', activation='sigmoid')(x)

model = Model(inputs=Inputs, outputs=predictions)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
model.save('testLSTM.h5')

##Training model
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss'
    , factor=0.2
    , patience=2
    , min_lr=0.001
    )

history = model.fit(vInput, vLabel
    , batch_size=128
    , epochs=2
    , verbose=1
    , callbacks=[reduce_lr]
    , validation_split=0.2
    )

scores = model.evaluate(vInput, vLabel)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("w: %.2f%%" % ((1-scores[1])*100))

# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("png/"+"plot_history.png")
plt.clf()

