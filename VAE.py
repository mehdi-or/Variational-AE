#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:17:23 2020

This version is with Variational Auto Encoder
@author: sorouji
"""

from numpy.random import seed
seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)
import IPython as IP
IP.get_ipython().magic('reset -f')
import numpy as np
import h5py
from keras import optimizers
from keras.layers import Input, Dense, concatenate, Dropout, Lambda
from keras.utils import to_categorical
#import pydotplus
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
import os
from scipy import stats
from keras import backend as K
from sklearn.cross_decomposition import CCA
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV
#keras.utils.vis_utils.pydotplus = pydotplus
from keras.utils.vis_utils import plot_model
start = time.time()
#%reset -f

#%%
# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#%% DNN architechture
drop=0.2
# network parameters
#input_shape = (X.shape[1],)
#intermediate_dim = 512
#batch_size = 128
encoding_dim=20
# latent_dim = 2 , same as encoding dim
#epochs = 50
def nna(encoding_dim):
      # this is our input placeholder for VT (creating a tensor)
      #imputs is the same as decoding_vox
      decoding_vox = Input(shape=(X.shape[1],), name="VT-input")
      # this is our input placeholder for PFC (creating a tensor)
      decoding_vox2 = Input(shape=(y.shape[1],), name="PFC-input")
      # "encoded" is the encoded representation of the input
      encoded_l = Dense(1024, activation='tanh', name="encoded_VT1")
      encoded = encoded_l(decoding_vox)
      encoded= Dropout(drop)(encoded)
      
      encoded2_l = Dense(1024, activation='tanh', name="encoded_PFC1")
      encoded2 = encoded2_l(decoding_vox2)
      encoded2= Dropout(drop)(encoded2)
      
      encoded = Dense(512, activation='tanh', name="encoded_VT2")(encoded)
      encoded= Dropout(drop)(encoded)

      encoded2 = Dense(512, activation='tanh', name="encoded_PFC2")(encoded2)
      encoded2= Dropout(drop)(encoded2)
      encodedc = concatenate([encoded, encoded2], axis=-1)
      
      encodedc = Dense(512, activation='tanh', name="shared1")(encodedc)
      encodedc= Dropout(drop)(encodedc)
      
      z_mean = Dense(encoding_dim, name='z_mean')(encodedc)
      z_log_var = Dense(encoding_dim, name='z_log_var')(encodedc)
      '''
      shared_l = Dense(encoding_dim, activation='linear', name="shared2")
      shared = shared_l(encodedc)
      #shared = Dropout(drop)(shared)
      '''
      z = Lambda(sampling, output_shape=(encoding_dim,), name='z')([z_mean, z_log_var])
      # "decoded" is the lossy reconstruction of the input
      decoded = Dense(512, activation='tanh', name="decoded-VT1")(shared)
      decoded= Dropout(drop)(decoded)
      
      decoded2 = Dense(512, activation='tanh', name="decoded-PFC1")(shared)
      decoded2= Dropout(drop)(decoded2)
      
      classifier = Dense(40, activation='softmax', name='classifier3')(shared)
      #classifier = Dense(200, activation='tanh', name='classifier1')(shared)
      #classifier = Dropout(drop)(classifier)
      
      decoded = Dense(1024, activation='tanh', name="decoded-VT2")(decoded)
      decoded= Dropout(drop)(decoded)
      
      decoded2 = Dense(1024, activation='tanh', name="decoded-PFC2")(decoded2)
      decoded2= Dropout(drop)(decoded2)
      
      #classifier = Dense(128, activation='tanh', name='classifier2')(classifier)
      #classifier = Dropout(drop)(classifier)
      
      decoded = Dense(y.shape[1], activation='linear', name="decoded_PFC")(decoded)
      decoded2 = Dense(X.shape[1], activation='linear', name="decoded_VT")(decoded2)
      #classifier = Dense(40, activation='softmax', name='classifier3')(classifier)
      
      encoder = Model([decoding_vox, decoding_vox2], shared)
      decoder = Model([decoding_vox, decoding_vox2], [decoded, decoded2, classifier]) # model architecture
      plot_model(decoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
      
      # modeling the bottle neck of the autoencoder VT to VT
      #encoder = Model([decoding_vox, decoding_vox2,shared], [decoding_vox, decoding_vox2, zz])
      #SVG(model_to_dot(decoder).create( prog='dot', format='svg'))
      #plot_model(decoder, to_file='model_plot.png', show_shapes=True)
    
      # fitting the model and defining loss functiojn for each imput
      opt = optimizers.adam(lr = 1e-4)
      decoder.compile(opt, loss=['mean_squared_error', 'mean_squared_error', 'categorical_crossentropy'],
                      loss_weights=[1, 1, 0.05], metrics={'classifier3': 'accuracy','decoded_PFC' : 'mse', 'decoded_VT' : 'mse'})
      return decoder, encoder, shared_l, encoded_l, encoded2_l
  
#%% Function for Reading the file
def importData (person):

    with h5py.File(r'/home/sorouji/Desktop/File/hrfAll_DLPFC_PETERS.hdf5','r') as hdf:
        #ls = list(hdf.keys())
        data1 = hdf.get('items/'+str(person)+'/rcargs/items/0')
        y0 = np.array(data1)
    
    with h5py.File(r'/home/sorouji/Desktop/File/hrfAll_VT_PETERS.hdf5','r') as hdf:
        #ls = list(hdf.keys())
        data2 = hdf.get('items/'+str(person)+'/rcargs/items/0')
        X0 = np.array(data2)
    
    y = np.delete(y0, np.where(~y0.any(axis=0))[0], axis=1)  # deleting the zero features (which are in columns)
    X = np.delete(X0, np.where(~X0.any(axis=0))[0], axis=1)  # deleting the zero features (which are in columns)
    '''
    #Winsorizing
    #Lower and uper limits of each feature in VT and PFC
    outlier_lower_X = np.median(X, axis=0)-1.5*stats.iqr(X, axis=0)
    outlier_upper_X = np.median(X, axis=0)+1.5*stats.iqr(X, axis=0)
    outlier_lower_y = np.median(y, axis=0)-1.5*stats.iqr(y, axis=0)
    outlier_upper_y = np.median(y, axis=0)+1.5*stats.iqr(y, axis=0)
    
    #removing outliers in VT and PFC
    for i in range(X.shape[1]):
       X[X[:,i]<outlier_lower_X[i],i]=outlier_lower_X[i]
       X[X[:,i]>outlier_upper_X[i],i]=outlier_upper_X[i]       
    for i in range(y.shape[1]):
       y[y[:,i]<outlier_lower_y[i],i]=outlier_lower_y[i]
       y[y[:,i]>outlier_upper_y[i],i]=outlier_upper_y[i] 
    '''

    
    #indices=X<outlier_lower_X.T    
    y = stats.zscore(y, axis=0)
    X = stats.zscore(X, axis=0)

    labels00 = pd.read_excel('/home/sorouji/Desktop/File/label.xlsx')
    labels0 = labels00.values
    #classes0 = np.unique(labels00)
    print('Data is loaded: \n')
    classes = np.unique(labels0)  # finding the number classes, here is 40
    class_number = np.arange(len(classes))
    map2labels = np.arange(labels0.shape[0]) # map to restore the correct order of lables

    # removing the contaminated data
    remove=np.zeros((len(labels0)))
    for i in range(len(labels0)-1):
        if labels0[i]!=labels0[i+1]:
                remove[i] = 1 #removing first contamination (last expl from the 1st mini-block)
                #remove[i+1] = 1 #removing second contamination
    remove=np.array(np.where(remove == 1))
    y = np.delete(y, remove, axis=0)
    X = np.delete(X, remove, axis=0)
    labels = np.delete(labels0, remove, axis=0) #removing contaminated data from labels
    map2 = np.delete(map2labels, remove, axis=0) #updating the map for the labels
    return X, y, labels, class_number, map2, labels0, classes

#%% remaping
def remapping (map2_test, FirstGuess):
    actual_test_class = labels0[map2_test] # to find the order of the test-set labels after shuffling
    #finding the index of the max value in each row which denotes the number of the class
    col_index1 = np.argmax(FirstGuess, axis=1)
    predicted_class = classes[col_index1] #mapping to clss(first guess)
    # Just a reshape to (n,1)
    predicted_class = np.reshape(predicted_class, (len(predicted_class),1)) 
    score1 = FirstGuess[np.arange(len(col_index1)), col_index1] #first guess score
    # just a reshpae
    score1 = np.reshape(score1, (len(score1),1))
    SecondGuess = np.zeros(FirstGuess.shape)
    SecondGuess = list (FirstGuess) # To creat a compeletly new copy instead of using the same pointer
    SecondGuess = np.asarray(SecondGuess)
    SecondGuess[np.arange(len(col_index1)), col_index1]= 0
    col_index2 = np.argmax(SecondGuess, axis=1)
    predicted_class2 = np.reshape(classes[col_index2], (len(classes[col_index2]),1))
    score2 = SecondGuess[np.arange(len(col_index2)), col_index2] #second guess score
    score2 = np.reshape(score2, (len(score1),1))
    map2_test = np.reshape(map2_test, (len(map2_test),1))
    actual_test_class = np.reshape(actual_test_class ,(len(actual_test_class),1))
    comparison = np.concatenate((actual_test_class, predicted_class, score1, \
                                 predicted_class2, score2, map2_test), axis=1)
    return comparison, predicted_class2


#%% assigning a number to each class, ie 0 for 'ant', etc
persons = range(37,38)
pers_num = len(persons)
for person in persons: 
    X, y, labels, class_number, map2, labels0, classes = importData (person)
    labels2num = np.zeros((1, len(labels)))
    for item in class_number:
        itemindex = np.array(np.where(labels == classes[item]))
        labels2num[0, itemindex[0, :]] = item
    
    labels2categ = to_categorical(labels2num[0, :], num_classes=len(class_number))
    # Splitting the dataset into the Training set and Test set
    #if the code is not working comment out this section
    X_train, X_test, y_train, y_test, label_train, label_test, map2_train, map2_test = train_test_split(
            X, y, labels2categ, map2, test_size=0.1, random_state=1)
    X_train, X_val, y_train, y_val, label_train, label_val, map2_train, map2_val = train_test_split(
            X_train, y_train, label_train, map2_train, test_size=0.2, random_state=1)
    
    # Feature Scaling----- this method scales things a little bit differently than my method
    scx = MinMaxScaler(feature_range = (-1, 1))
    scy = MinMaxScaler(feature_range = (-1, 1))
                
    X_train_scaled = scx.fit_transform(X_train)
    X_val_scaled = scx.transform(X_val)
    X_test_scaled = scx.transform(X_test)
        
    y_train_scaled = scy.fit_transform(y_train)
    y_val_scaled = scy.transform(y_val)
    y_test_scaled = scy.transform(y_test)

    decoder, encoder, shared, encoder1, encoder2 = nna(encoding_dim)
    decoder.fit([X_train, y_train], [y_train, X_train, label_train], epochs=200, shuffle=True,
                validation_data=([X_val, y_val], [y_val, X_val, label_val]))
        
    intermediate_layer_model = Model(inputs=decoder.input, outputs=decoder.get_layer('shared2').output)
    
    # Predicting VT, PFC, and classes
    [decoded_PFC, decoded_VT, FirstGuess] = decoder.predict([X_test, y_test])
    BottleNeck = intermediate_layer_model.predict([X_test, y_test])
    test1 = labels0[map2_test]
    test_label = pd.DataFrame(test1, map2_test)
    
    comparison, predicted_class2 = remapping (map2_test, FirstGuess)
    
#%%

#from sklearn.metrics import r2_score
#DataFrame(result).to_csv("/home/sorouji/Desktop/UCI/Research/result32/data.csv")

correct = sum(comparison[:,0]==comparison[:,1])
accu=correct/labels0[map2_test][:,0].shape[0]

plt.figure(1)
plt.plot(y_test[1,:])
plt.plot(decoded_PFC[1,:])
plt.gca().legend(('PFC_actual','PFC_predicted'))

plt.figure(2)
plt.hist(y_test[1,:],100)
plt.hist(decoded_PFC[1,:],100)
plt.gca().legend(('PFC_actual','PFC_predicted'))

plt.figure(3)
plt.plot(decoded_PFC[1,:],y_test[1,:],'.')
[a, b] = np.polyfit(decoded_PFC[1,:],y_test[1,:],1)
line1=np.polyval([a, b], decoded_PFC[1,:])
plt.plot(decoded_PFC[1,:],line1)
plt.xlabel('Predicted PFC')
plt.ylabel('actual PFC')



rscore = np.corrcoef(y_test[1,:], decoded_PFC[1,:])

plot_model(decoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

end = time.time()
RunTime = end-start
print("Runtime is:", RunTime, "seconds")

"""
decoder = KerasClassifier(build_fn = nna)
#decoder = nna()

parameters = {'batch_size': [25, 32], 
              'epochs': [100, 200]}

grid_search=GridSearchCV(estimator = decoder,  param_grid = parameters, cv = 6)

# PFC to PFC
grid_search.fit([X_train, y_train], [X_train, y_train, label_train], shuffle=True)
#decoder.fit([x_train, y_train],[x_train, y_train, zz], epochs=1, shuffle=False, validation_data=None)
"""
#%%
rscore = np.zeros((67, 67, len(classes)))
for item in range(len(classes)):
    p=np.where(labels==classes[item])
    index=np.asarray(p[0])
    for row in range(len(index)):
        for col in range(len(index)):
            rscore[row, col, item] = np.corrcoef(X[index[row]], X[index[col]])[0][1]

oo=np.apply_over_axes(np.mean, rscore, 1)
oo=np.reshape(oo, (oo.shape[0], oo.shape[2]))

cca = CCA(n_components=20)

nam=cca.fit(X_test, y_test)
CCA_learned = nam.predict(X_test)
CCA_rscore = np.corrcoef(y_test[1,:], CCA_learned[1,:])
DataFrame(BottleNeck).to_csv("/home/sorouji/Desktop/UCI/Research/BottleNeck"+ str(person)+ ".csv")
plt.figure(3)
plt.plot(CCA_learned[1,:],y_test[1,:],'.')

plt.figure(1)
plt.plot(CCA_learned[1,:])
