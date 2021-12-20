from os import X_OK
import numpy as np

import tensorflow as tf

#['%', '(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'times']

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
    path='mnist.npz'
)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape(*x_train.shape[:-2], -1)
x_test = x_test.reshape(*x_test.shape[:-2], -1)


x = np.concatenate((x_train, x_test), axis = 0)
print(x.shape)


np.save('C:\\Users\\klara\\Desktop\\mnist_x', x_train)
np.save('C:\\Users\\klara\\Desktop\\mnist_y', y_train)

print (x_train.shape, y_train.shape)