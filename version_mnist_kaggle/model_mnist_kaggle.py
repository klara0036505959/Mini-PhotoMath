import keras
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LeakyReLU
)

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import sys
import argparse

def load_data(class_labels, data_name, label_name, train=0.85, val=0.15, one_hot_enc = False):
    '''Function to Load data from .npy files and split them into training and validation sets '''
    data = pd.DataFrame( np.load( data_name ) )
    labels = pd.DataFrame( np.load( label_name ) )
    
    #data = np.load(data_name)
    #labels = np.load(label_name)
    

    assert data.shape[0] == labels.shape[0]
    assert isinstance(train, float)
    isinstance(val, float), "train and val must be of type float, not {0} and {1}".format(type(train), type(val))
    assert ((train + val) == 1.0), "train + val must equal 1.0"

    
    sidx = int(data.shape[0]*train)
    _data  = {'train': data.iloc[:sidx].to_numpy(),   'val': data.iloc[sidx+1:].to_numpy()}
    if not one_hot_enc:
        _labels= {'train': labels.iloc[:sidx,:].to_numpy(), 'val': labels.iloc[sidx+1:,:].to_numpy()}
    else:
        labels = labels.rename(columns = {0:'labels'})
        labels = labels.astype({"labels": str})
        #print(labels)
        labels['labels'] = labels['labels'].map(class_labels)
        labels['labels'] = labels['labels'].astype(pd.CategoricalDtype(categories=range(16)))
        one_hot = pd.get_dummies(labels['labels'])
        #print(one_hot)
        _labels= {'train': one_hot.iloc[:sidx,:].to_numpy(), 'val': one_hot.iloc[sidx+1:,:].to_numpy()}
        #print(_labels['train'].shape)

    assert (_data['train'].shape[0] == _labels['train'].shape[0])
    assert (_data['val'].shape[0] == _labels['val'].shape[0])
    return _data, _labels

# Creating dictionary of labels
#class_labels = {str(x):x for x in range(10)}
#class_labels.update({'+':10, 'times':11, '-':12 })
#label_class = dict( zip(class_labels.values(), class_labels.keys() ))

class_labels = {'%': 0, '(': 1, ')': 2, '+': 3, '-': 4, '0': 5, '1': 6, '2': 7, '3': 8,
 '4': 9, '5': 10, '6': 11, '7': 12, '8': 13, '9': 14, 'times': 15}

#inv_map = {v: k for k, v in class_labels.items()}
 


# Loading data from .npy file and spliting into training and validation sets
path = 'datasets_npy\\'
data = {"train": 0, "val":0}
labels = {"train": 0, "val":0}

path1 = path+'x_kaggle.npy'
path2 = path+'y_kaggle.npy'

data1, labels1 = load_data(class_labels, path1 , path2, train = 0.85 , val = 0.15)

path1 = path+'mnist_x.npy'
path2 = path+'mnist_y.npy'

data2, labels2 = load_data(class_labels, path1 , path2, train = 0.85 , val = 0.15, one_hot_enc=True)

data['train'] = np.concatenate((data1["train"], data2["train"]), axis = 0)
labels['train'] = np.concatenate((labels1["train"], labels2["train"]), axis = 0)

data['val'] = np.concatenate((data1["val"], data2["val"]), axis = 0)
labels['val'] = np.concatenate((labels1["val"], labels2["val"]), axis = 0)

'''
np.save('datasets_npy\\train_x', data['train'])
np.save('datasets_npy\\val_x', data['val'])
np.save('datasets_npy\\train_y', labels['train'])
np.save('datasets_npy\\val_y', labels['val'])
'''

img_x, img_y = 28, 28
x_train = data['train'].reshape(data['train'].shape[0], img_x, img_y, 1)
y_train = labels['train']
x_test = data['val'].reshape(data['val'].shape[0], img_x, img_y, 1)
y_test = labels['val']
input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
num_classes = y_train.shape[1]

model = Sequential()
model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1),input_shape=input_shape))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1)  )
model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1)  )
model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1)  )
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1000 ))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(256 ))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(num_classes, activation='softmax'))

adam1 = tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam1,metrics=['accuracy'] )

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint
filepath="checkpoint_new_lr.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(x_train, y_train, epochs=40, batch_size=128,validation_split=0, validation_data=(x_test, y_test), callbacks=callbacks_list)

model.save("model_cijeli.h5")

# serialize model to JSON
model_json = model.to_json()
with open("CNN_ver{0}.1.json".format(1), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("CNN_ver{0}.1.h5".format(1))
print("Saved model to disk")
