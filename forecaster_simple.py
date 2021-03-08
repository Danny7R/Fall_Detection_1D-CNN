import numpy as np
import pandas as pd
from time import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D, MaxPooling1D, Dropout, add
from tensorflow.keras.models import Model, load_model
import tsa
from data import load_data
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')


m = 50
t0 = time()
xN_1, xF_1, yN_1, yF_1 = load_data(m=m, d=1)
t1 = time()
print('data processing time: ', t1 - t0, '(s)')


def model1(input_shape):
    model_input = Input(input_shape)

    x = Conv1D(32, 3, strides=1, padding='same', kernel_initializer='he_normal', activation='relu')(model_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)
    # x = Dropout(0.2)(x)
    x = Conv1D(64, 3, strides=1, padding='same', kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)
    x = Conv1D(128, 3, strides=1, padding='same', kernel_initializer='he_normal', activation='relu')(x)
    # x = Dropout(0.2)(x)
    x = Flatten()(x)
    # x = Dense(64, activation='relu')(x)
    output = Dense(4, activation='linear')(x)
    model = Model(inputs=model_input, outputs=output, name='model1')
    return model


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


m1 = model1((m, 4))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
m1.compile(optimizer='adam', loss="mse", metrics=[rmse])

history = m1.fit(xN_1, yN_1, epochs=100, batch_size=512, validation_split=0.3, verbose=2)
m1.save('forecaster_7_m20_3cnn-1fc_rmse1.7%.h5')
