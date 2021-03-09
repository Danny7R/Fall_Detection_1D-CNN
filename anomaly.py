import numpy as np
# import pandas as pd
from time import time
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
from tensorflow.keras.utils import get_custom_objects
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, OneClassSVM
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import tsa
from data import load_data

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


get_custom_objects().clear()
get_custom_objects()["rmse"] = rmse

m = 50
t0 = time()
xN_1, xF_1, yN_1, yF_1 = load_data(m=m, d=1)
t1 = time()
print('data processing time: ', t1 - t0, '(s)')

forecaster = load_model('forecaster_4_m50_3cnn_rmse1.4%.h5')

yhN_1 = forecaster.predict(xN_1, batch_size=np.power(2, 16), verbose=0, steps=None)
yhF_1 = forecaster.predict(xF_1, batch_size=np.power(2, 16), verbose=0, steps=None)

print('yh:', yhN_1.shape, yhF_1.shape)

xN_2 = np.concatenate((yhN_1, yN_1, np.std(xN_1, axis=1), np.mean(xN_1, axis=1),
                       np.percentile(xN_1, 25, axis=1), np.percentile(xN_1, 75, axis=1),
                       np.max(xN_1, axis=1), np.min(xN_1, axis=1)), axis=1)

xF_2 = np.concatenate((yhF_1, yF_1, np.std(xF_1, axis=1), np.mean(xF_1, axis=1),
                       np.percentile(xF_1, 25, axis=1), np.percentile(xF_1, 75, axis=1),
                       np.max(xF_1, axis=1), np.min(xF_1, axis=1)), axis=1)

print(xN_2.shape, xF_2.shape)

x2 = np.concatenate((xN_2, xF_2), axis=0)
y2 = np.concatenate((np.zeros(xN_2.shape[0]), np.ones(xF_2.shape[0])), axis=0)
print(x2.shape, y2.shape)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2)
print(x2_train.shape, x2_test.shape, y2_train.shape, y2_test.shape)

fall_detector = RandomForestClassifier(n_estimators=400, n_jobs=-1)  # max_depth=20, min_samples_split=5
# fall_detector = SVC(C=50.0, gamma=20)
# fall_detector = MLPClassifier(hidden_layer_sizes=(100, 50))

fall_detector = fall_detector.fit(x2_train, y2_train)
print('fall_detector train:', fall_detector.score(x2_train, y2_train))
print('fall_detector test', fall_detector.score(x2_test, y2_test))

y_pred = fall_detector.predict(x2_test)
cm = confusion_matrix(y2_test, y_pred)


# fall_detector2 = OneClassSVM(nu=0.001, kernel='rbf', gamma=0.1)
# fall_detector2 = IsolationForest(n_estimators=400, n_jobs=-1)  # , max_samples=100000, contamination=0.001, max_features=1.0
# y_pred_train = fall_detector2.fit_predict(x2_train[y2_train == 0])
# print('training accuracy: ', np.sum(y_pred_train == 1)/np.sum(y2_train == 0))

# y_pred_test = fall_detector2.predict(x2_test)
# y2_tst = -2*y2_test + 1
# print('test accuracy: ', np.sum(y_pred_test == y2_tst) / x2_test.shape[0])

# cm = confusion_matrix(-1*y2_tst, -1*y_pred_test)

print('Confusion Matrix : \n', cm)

total1 = np.sum(np.sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total1
print('Accuracy : ', accuracy)
TPR = cm[0, 0] / (cm[0, 0] + cm[0, 1])
FPR = cm[0, 1] / (cm[0, 1] + cm[1, 1])
print('TPR : ', TPR)
print('FPR : ', FPR)

### saving and loading the model
from joblib import dump, load
dump(fall_detector, 'rforest_32input_400_99.66%_25th.joblib')

fall_detector = load('rforest_32input_400_99.66%_25th.joblib')

# import pickle
# pickle.dump(fall_detector, open('finalized_model.sav', 'wb'))
# loaded_model = pickle.load(open(filename, 'rb'))
