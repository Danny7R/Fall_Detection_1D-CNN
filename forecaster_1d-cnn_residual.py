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


def residual(layer_in, n_filters):
    merge_input = layer_in
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv1D(n_filters, 1, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)

    conv1 = Conv1D(n_filters, 3, strides=1, padding='same', activation='linear', kernel_initializer='he_normal')(layer_in)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv1D(n_filters, 3, strides=1, padding='same', activation='linear', kernel_initializer='he_normal')(conv1)

    layer_out = add([conv2, merge_input])
    layer_out = Activation('relu')(layer_out)
    return layer_out


def model4(input_shape):
    model_input = Input(input_shape)

    x = residual(model_input, 64)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)
    x = Dropout(0.2)(x)
    x = residual(x, 128)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)
    x = Conv1D(128, 3, strides=1, padding='same', kernel_initializer='he_normal', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(4, activation='linear')(x)
    model = Model(inputs=model_input, outputs=output, name='model4')
    return model


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


m1 = model4((m, 4))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
m1.compile(optimizer='adam', loss="mse", metrics=[rmse])

history = m1.fit(xN_1, yN_1, epochs=100, batch_size=200, validation_split=0.3, verbose=2)
m1.save('forecaster_8_m50.h5')
