# import os
# import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tsa
from time import time


def load_data(m=30, d=1):

    df = pd.read_csv("ActivitySampleData.csv", header=0, sep=',')  # , delim_whitespace=True
    # correct = df.apply(lambda x: x.category in x.category_group, axis=1)
    # correct = [x[0] in x[1] for x in zip(df['category'], df['category_group'])]  # faster
    # df = df[correct]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df[['x', 'y', 'z', 'acceleration']].values)

    FALL = df['category_group'].str.contains('Front|Left|Right|Back|Fall', case=False)
    dfF = df[FALL]
    dfN = df[~FALL]
    # gp = df.groupby('category')

    xF_total = np.array([]).reshape((0, m, 4))
    yF_total = np.array([]).reshape((0, 4))
    xN_total = np.array([]).reshape((0, m, 4))
    yN_total = np.array([]).reshape((0, 4))

    fall_ids = dfF.category_group.unique()
    for i in fall_ids:
        dfid = dfF[dfF['category_group'] == i]
        dfid = dfid[['x', 'y', 'z', 'acceleration']].dropna()
        tseries = dfid.values.astype('float32')  # shape: n_samples * n_features
        tseries = scaler.transform(tseries)
        x, y = tsa.delay_embed2(tseries, m, d)
        xF_total = np.concatenate((xF_total, x), axis=0, out=None)
        yF_total = np.concatenate((yF_total, y), axis=0, out=None)

    print('fall: ', xF_total.shape, yF_total.shape)

    nofall_ids = dfN.category_group.unique()
    for i in nofall_ids:
        dfid = dfN[dfN['category_group'] == i]
        dfid = dfid[['x', 'y', 'z', 'acceleration']].dropna()
        tseries = dfid.values.astype('float32')  # shape: n_samples * n_features
        tseries = scaler.transform(tseries)
        x, y = tsa.delay_embed2(tseries, m, d)
        xN_total = np.concatenate((xN_total, x), axis=0, out=None)
        yN_total = np.concatenate((yN_total, y), axis=0, out=None)

    print('no fall: ', xN_total.shape, yN_total.shape)

    return xN_total, xF_total, yN_total, yF_total


if __name__ == '__main__':

    t0 = time()
    xN_total, xF_total, yN_total, yF_total = load_data(m=30, d=1)
    t1 = time()
    print('data processing time: ', t1 - t0, '(s)')
