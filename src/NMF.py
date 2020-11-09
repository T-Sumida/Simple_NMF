# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple


class NMF():
    """
    簡単なNMFを行うクラス
    """

    def setParam(self, k: int, row: int, column: int):
        """NMFのパラメータ設定

        Args:
            k (int): 因子数
            row (int): 列数
            column (int): 行数
        """
        self.__k = k
        self.__row = row
        self.__column = column

        self.__dictionary = np.random.random_sample([self.__row, self.__k])
        self.__activation = np.random.random_sample([self.__k, self.__column])

    def setDictionary(self, index: int, data: List):
        """辞書行列へのデータ設定
        Args:
            index (int): 因子インデックス ( 0 <= index < k)
            data (List): データ
        """
        if index >= self.__k and len(data) != self.__row:
            print("Please NMF.setParam(k,row,column)")
            print(f"k = {self.__k}")
            print(f"row = {self.__row}")
            return

        self.__dictionary[:, index] = np.array(data[:self.__row], np.float32)

    def setAnalyzData(self, data: List, k: int):
        """分解対象行列データを登録

        Args:
            data (List): 分解対象行列
            k (int): 因子数
        """
        if len(np.shape(data)) == 1:
            self.__data = np.ones([np.shape(data)[0], 1], np.float32)
            self.setParam(k, np.shape(data)[0], 1)
        else:
            self.__data = data
            self.setParam(k, np.shape(data)[0], np.shape(data)[1])

    def separate_euc_with_template(self, iter: int = 200) -> Tuple[np.array, np.array]:
        """テンプレートありのEUC-divergence仕様の分離処理

        Args:
            iter (int, optional): 反復更新回数. Defaults to 200.

        Returns:
            Tuple[np.array, np.array]: [辞書行列, 励起行列]
        """
        counter = 0

        while counter < iter:
            approx = np.dot(self.__dictionary, self.__activation)

            wh = np.dot(np.transpose(self.__dictionary), self.__data)
            wt = np.dot(np.transpose(self.__dictionary), approx)

            bias = wh/wt
            bias[np.isnan(bias)] = 0

            self.__activation = self.__activation * bias
            counter += 1
        return self.__dictionary, self.__activation

    def separate_kl_with_template(self, iter: int = 200) -> Tuple[np.array, np.array]:
        """テンプレートありのKL-divergence仕様の分離処理

        Args:
            iter (int, optional): 反復更新回数. Defaults to 200.

        Returns:
            Tuple[np.array, np.array]: [辞書行列, 励起行列]
        """
        counter = 0
        while counter < iter:
            approx = np.dot(self.__dictionary, self.__activation)

            w = self.__data/approx
            w[np.isnan(w)] = 0
            wh = np.dot(np.transpose(self.__dictionary), w)

            wt = np.ones([1, self.__k], np.float32)
            wt[:] = sum(self.__dictionary[:, :])
            wt = np.transpose(wt)

            bias = wh/wt
            bias[np.isnan(bias)] = 0

            self.__activation = self.__activation * bias
            counter += 1
        return self.__dictionary, self.__activation

    def separate_is_with_template(self, iter: int = 200) -> Tuple[np.array, np.array]:
        """テンプレートありのIS-divergence仕様の分離処理

        Args:
            iter (int, optional): 反復更新回数. Defaults to 200.

        Returns:
            Tuple[np.array, np.array]: [辞書行列, 励起行列]
        """
        counter = 0

        while counter < iter:
            approx = np.dot(self.__dictionary, self.__activation)
            wt = np.ones([1, self.__k], np.float32)

            w1 = self.__data/approx
            w2 = np.transpose(self.__dictionary)/sum(np.transpose(approx[:]))

            w1[np.isnan(w1)] = 0
            w2[np.isnan(w2)] = 0

            wh = np.dot(w2, w1)
            wt[:] = sum(np.transpose(w2[:]))
            wt = np.transpose(wt)

            bias = wh/wt
            bias[np.isnan(bias)] = 0

            self.__activation = self.__activation * np.sqrt(bias)
            counter += 1

        return self.__dictionary, self.__activation

    def separate_euc_without_template(self, iter: int = 200) -> Tuple[np.array, np.array]:
        """テンプレートなしのEUC-divergence仕様の分離処理

        Args:
            iter (int, optional): 反復更新回数. Defaults to 200.

        Returns:
            Tuple[np.array, np.array]: [辞書行列, 励起行列]
        """
        counter = 0

        while counter < iter:
            approx = np.dot(self.__dictionary, self.__activation)

            wh = np.dot(np.transpose(self.__dictionary), self.__data)
            wt = np.dot(np.transpose(self.__dictionary), approx)

            bias = wh/wt
            bias[np.isnan(bias)] = 0

            self.__activation = self.__activation * bias

            approx = np.dot(self.__dictionary, self.__activation)
            wh = np.dot(self.__data, np.transpose(self.__activation))
            wt = np.dot(approx, np.transpose(self.__activation))

            bias = wh/wt
            bias[np.isnan(bias)] = 0
            self.__dictionary = self.__dictionary * bias
            counter += 1

        return self.__dictionary, self.__activation

    def separate_kl_without_template(self, iter: int = 200) -> Tuple[np.array, np.array]:
        """テンプレートなしのKL-divergence仕様の分離処理

        Args:
            iter (int, optional): 反復更新回数. Defaults to 200.

        Returns:
            Tuple[np.array, np.array]: [辞書行列, 励起行列]
        """
        counter = 0
        while counter < iter:
            approx = np.dot(self.__dictionary, self.__activation)

            w = self.__data/approx
            w[np.isnan(w)] = 0
            wh = np.dot(np.transpose(self.__dictionary), w)

            wt = np.ones([1, self.__k], np.float32)
            wt[:] = sum(self.__dictionary[:, :])
            wt = np.transpose(wt)

            bias = wh/wt
            bias[np.isnan(bias)] = 0

            self.__activation = self.__activation * bias

            approx = np.dot(self.__dictionary, self.__activation)
            w = self.__data/approx
            w[np.isnan(w)] = 0
            wh = np.dot(w, np.transpose(self.__activation))

            wt = np.ones([self.__k, 1], np.float32)
            wt = sum(np.transpose(self.__activation[:]))
            wt = np.transpose(wt)

            bias = wh/wt
            self.__dictionary = self.__dictionary * bias
            counter += 1
        return self.__dictionary, self.__activation

    def separate_is_without_template(self, iter: int = 200) -> Tuple[np.array, np.array]:
        """テンプレートなしのIS-divergence仕様の分離処理

        Args:
            iter (int, optional): 反復更新回数. Defaults to 200.

        Returns:
            Tuple[np.array, np.array]: [辞書行列, 励起行列]
        """
        counter = 0

        while counter < iter:
            approx = np.dot(self.__dictionary, self.__activation)
            wt = np.ones([1, self.__k], np.float32)

            w1 = self.__data/approx
            w2 = np.transpose(self.__dictionary)/sum(np.transpose(approx[:]))
            w1[np.isnan(w1)] = 0
            w2[np.isnan(w2)] = 0

            wh = np.dot(w2, w1)
            wt[:] = sum(np.transpose(w2[:]))
            wt = np.transpose(wt)

            bias = wh/wt
            bias[np.isnan(bias)] = 0

            self.__activation = self.__activation * np.sqrt(bias)

            approx = np.dot(self.__dictionary, self.__activation)
            w1 = self.__data/approx
            w2 = self.__activation/sum(approx[:])
            w1[np.isnan(w1)] = 0
            w2[np.isnan(w2)] = 0

            wh = np.dot(w1, np.transpose(w2))
            wt = sum(np.transpose(w2[:]))

            bias = wh/wt
            bias[np.isnan(bias)] = 0
            self.__dictionary = self.__dictionary * np.sqrt(bias)

            counter += 1

        return self.__dictionary, self.__activation
