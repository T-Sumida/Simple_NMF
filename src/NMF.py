# coding:utf-8

import numpy as np
import Update


class NMF():
    """
    簡単なNMFを行うクラス
    """



    def setParam(self, k, row, column):
        """
        :param k: 因子分回数
        :param row: 列
        :param column: 行
        :return:
        """
        self.__k = k
        self.__row = row
        self.__column = column

        self.__dictionary = np.random.random_sample([self.__row, self.__k])
        self.__activation = np.random.random_sample([self.__k, self.__column])

    def setDictionary(self, index, data):
        """
        基本的にテンプレートありのNMFなので、テンプレートをセットしてください（デフォルトは乱数）
        :param index: どのテンプレートかのindex (0 <= index < k)
        :param data: テンプレートデータ
        :return:
        """
        if index >= self.__k and len(data) != self.__row:
            print "Please NMF.setParam(k,row,column)"
            print "k = " + str(self.__k)
            print "row = " + str(self.__row)
            return

        self.__dictionary[:, index] = np.array(data[:self.__row], np.float32)

    def setAnalyzData(self, data, k):
        """
        一番最初に、このメソッドを呼んでください
        :param data: 分解するデータ
        :param k: 因子分解数
        :return:
        """
        if len(np.shape(data)) == 1:
            self.__data = np.ones([np.shape(data)[0], 1], np.float32)
            self.setParam(k, np.shape(data)[0], 1)
        else:
            self.__data = data
            self.setParam(k, np.shape(data)[0], np.shape(data)[1])

    def start(self, algf=Update.euc,iter=10):
        """
        因子分解を開始する
        :param algf: Updateファイルの中にあるどのアルゴリズムかを選ぶ
        :param iter: 反復更新回数
        :return: 更新後の辞書行列と励起行列
        """
        self.__data,self.__activation = algf(self.__data,self.__dictionary,self.__activation,iter)

        return self.__dictionary, self.__activation


