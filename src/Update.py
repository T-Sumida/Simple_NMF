#coding: utf-8
import numpy as np

def euc(data,dictionary,activation,iter):
    """
    EUCに基づいた乗法更新式を実行する関数
    :param data: 元行列
    :param dictionary: 辞書行列
    :param activation: 励起行列
    :param iter: 反復更新回数
    :return: 更新後の辞書行列と励起行列
    """
    counter = 0

    while counter < iter:
        approx = np.dot(dictionary , activation)

        wh = np.dot(np.transpose(data) , dictionary)
        wt = np.dot(np.transpose(approx) , dictionary)

        bias = wh/wt
        bias[np.isnan(bias)] = 0

        activation = activation * np.transpose(bias)
        counter += 1

    return dictionary,activation



def k_l(data,dictionary,activation,iter):
    """
    KLに基づいた乗法更新式を実行する関数
    :param data: 元行列
    :param dictionary: 辞書行列
    :param activation: 励起行列
    :param iter: 反復更新回数
    :return: 更新後の辞書行列と励起行列
    """

    counter = 0
    while counter < iter:
        approx = np.dot(dictionary , activation)

        w = data/approx
        w[np.isnan(w)] = 0
        wh = np.dot(np.transpose(w),dictionary)

        wt = sum(dictionary[:,:])


        bias = wh/wt
        bias[np.isnan(bias)] = 0

        activation = activation * np.transpose(bias)
        counter += 1

    return dictionary,activation



def i_s(data,dictionary,activation,iter):
    """
    ISに基づいた乗法更新式を実行する関数
    :param data: 元行列
    :param dictionary: 辞書行列
    :param activation: 励起行列
    :param iter: 反復更新回数
    :return: 更新後の辞書行列と励起行列
    """
    counter = 0

    while counter < iter:
        approx = np.dot(dictionary , activation)

        w1 = np.transpose(data/approx)
        w2 = np.transpose(np.transpose(dictionary)/np.transpose(sum(np.transpose(approx[:]))))
        w1[np.isnan(w1)] = 0
        w2[np.isnan(w2)] = 0

        wh = np.dot(w1, w2)
        wt = sum(w2[:])


        bias = wh/wt
        bias[np.isnan(bias)] = 0

        activation = activation * np.transpose(bias)
        counter += 1

    return dictionary,activation