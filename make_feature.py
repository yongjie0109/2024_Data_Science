import numpy as np

def make_feature(data):
    feature = np.zeros(5)
    feature[0] = cal_point(data)
    feature[1] = cal_std(data)
    feature[2] = cal_width(data)
    feature[3], feature[4] = cal_cr(data)
    return feature

def cal_point(data):
    return data.shape[0]

def cal_std(data):
    n = cal_point(data)
    m = np.mean(data, axis=0)
    x = data[:, 0]
    y = data[:, 1]
    sigma = np.sqrt(1/n * (np.sum((x - m[0])**2) + np.sum((y - m[1])**2)))
    return sigma

def cal_width(data):
    width = np.sqrt((data[0, 0] - data[-1, 0])**2 + (data[0, 1] - data[-1, 1])**2)
    return width

def cal_cr(data):
    n = cal_point(data)
    x = data[:, 0]
    y = data[:, 1]
    A = np.vstack([-2 * x, -2 * y, np.ones(n)]).T
    b = -x**2 - y**2

    x_p = np.linalg.lstsq(A, b, rcond=None)[0]
    xc = x_p[0]
    yc = x_p[1]
    radius = np.sqrt(xc**2 + yc**2 - x_p[2])
    circularity = np.sum((radius - np.sqrt((xc - x)**2 + (yc - y)**2))**2)
    return circularity, radius