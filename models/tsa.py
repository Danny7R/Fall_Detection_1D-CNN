import numpy as np


def delay_embed(ds, m, d):
    w = m+1
    data = ds[(np.arange(w)*(d)) + np.arange(np.max(ds.shape[0] - (w-1)*(d), 0)).reshape(-1, 1)]
    return np.squeeze(data)


def multi_delay_embed(ds, m, d):
    total = np.expand_dims(delay_embed(ds[:, 0], m, d), axis=2)
    for i in range(1, ds.shape[1]):
        series_i = np.expand_dims(delay_embed(ds[:, i], m, d), axis=2)
        total = np.concatenate((total, series_i), axis=2)

    x = total[:, :-1, :]
    y = total[:, -1, :]
    return x, y


def delay_embed2(ds, m, d):
    w = m+1
    data = ds[(np.arange(w)*(d)) + np.arange(np.max(ds.shape[0] - (w-1)*(d), 0)).reshape(-1, 1)]
    data = np.squeeze(data)
    x = data[:, :-1, :]
    y = data[:, -1, :]
    return x, y


if __name__ == '__main__':

    ds = np.transpose(np.arange(15).reshape(3, 5))
    x_1, y_1 = multi_delay_embed(ds, m=3, d=1)
    x_2, y_2 = delay_embed2(ds, m=3, d=1)
    print(np.array_equal(x_1, x_2), np.array_equal(y_1, y_2))
