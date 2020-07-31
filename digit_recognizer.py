# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_sample = pd.read_csv('sample_submission.csv')
X = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])
X = np.reshape(X,(-1,28,28,1))
classes = np.unique(y)
num = len(classes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 42)

classifier = Sequential()

classifier.add(Convolution2D(32, (3,3), input_shape = (28, 28, 1), 
                             activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(64, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(128, (3,3), activation = 'relu',
                             padding = 'same'))
classifier.add(MaxPooling2D(pool_size = (1, 1)))
classifier.add(Convolution2D(512, (3,3), activation = 'relu', 
                             padding = 'same'))
classifier.add(MaxPooling2D(pool_size = (1, 1)))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dense(32, activation = 'relu'))
classifier.add(Dense(num, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])

batch_size = 32
epochs = 60
from keras.utils import to_categorical
y_train_one_hot = np.array(to_categorical(y_train))
y_test_one_hot = np.array(to_categorical(y_test))

train = classifier.fit(X_train, y_train_one_hot, batch_size = batch_size, 
                       epochs = epochs,verbose=1,
                       validation_data=(X_test, y_test_one_hot))

test_eval_train = classifier.evaluate(X_train, y_train_one_hot, verbose=0)
test_eval_test = classifier.evaluate(X_test, y_test_one_hot, verbose=0)
accuracy = {'train': test_eval_train[1], 'test': test_eval_test[1]}

X_pred = np.array(df_test.iloc[:,:])
X_pred = np.reshape(X_pred,(-1,28,28,1))
y_pred = classifier.predict(X_pred)
y_pred = np.argmax(np.round(y_pred),axis=1)

image_id = (np.arange(1,28001,dtype=int))

submission = pd.DataFrame(np.column_stack((image_id,y_pred)))
submission.columns = ['ImageId', 'Label']
submission.to_csv('Submission_DigitRecog_CNN.csv',index=False)

sub = pd.read_csv('Submission_DigitRecog_CNN.csv')












