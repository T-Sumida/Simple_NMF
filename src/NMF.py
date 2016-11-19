# coding:utf-8

import numpy as np


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


    def separate_euc_with_template(self,iter=200):
        """
        テンプレートありのEUC仕様の分離処理を行う
        :param iter:
        :return:
        """
        counter = 0

        while counter < iter:
            approx = np.dot(self.__dictionary , self.__activation)

            wh = np.dot(np.transpose(self.__dictionary) , self.__data)
            wt = np.dot(np.transpose(self.__dictionary) , approx)

            bias = wh/wt
            bias[np.isnan(bias)] = 0

            self.__activation = self.__activation * bias
            counter += 1
        return self.__dictionary,self.__activation


    def separate_kl_with_template(self,iter=200):
        """
        テンプレートありのKL仕様の分離処理を行う
        :param iter:
        :return:
        """
        counter = 0
        while counter < iter:
            approx = np.dot(self.__dictionary , self.__activation)

            w = self.__data/approx
            w[np.isnan(w)] = 0
            wh = np.dot(np.transpose(self.__dictionary),w)

            wt = np.ones([1, self.__k], np.float32)
            wt[:] = sum(self.__dictionary[:,:])
            wt = np.transpose(wt)


            bias = wh/wt
            bias[np.isnan(bias)] = 0

            self.__activation = self.__activation * bias
            counter += 1
        return self.__dictionary,self.__activation



    def separate_is_with_template(self,iter=200):
        """
        テンプレートありのIS仕様の分離処理を行う
        :param iter:
        :return:
        """
        counter = 0

        while counter < iter:
            approx = np.dot(self.__dictionary , self.__activation)
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

            self.__activation = self.__activation * bias
            counter += 1

        return self.__dictionary,self.__activation


    def separate_euc_without_template(self,iter=200):
        """
        テンプレートなしのEUC仕様の分離処理を行う
        :param iter:
        :return:
        """
        counter = 0

        while counter < iter:
            approx = np.dot(self.__dictionary , self.__activation)

            wh = np.dot(np.transpose(self.__dictionary) , self.__data)
            wt = np.dot(np.transpose(self.__dictionary) , approx)

            bias = wh/wt
            bias[np.isnan(bias)] = 0

            self.__activation = self.__activation * bias

            approx = np.dot(self.__dictionary,self.__activation)
            wh = np.dot(self.__data,np.transpose(self.__activation))
            wt = np.dot(approx,np.transpose(self.__activation))

            bias = wh/wt
            bias[np.isnan(bias)] = 0
            self.__dictionary = self.__dictionary * bias
            counter += 1

        return self.__dictionary,self.__activation


    def separate_kl_without_template(self,iter=200):
        """
        テンプレートなしのKL仕様の分離処理を行う
        :param iter:
        :return:
        """
        counter = 0
        while counter < iter:
            approx = np.dot(self.__dictionary , self.__activation)

            w = self.__data/approx
            w[np.isnan(w)] = 0
            wh = np.dot(np.transpose(self.__dictionary),w)

            wt = np.ones([1, self.__k], np.float32)
            wt[:] = sum(self.__dictionary[:,:])
            wt = np.transpose(wt)


            bias = wh/wt
            bias[np.isnan(bias)] = 0

            self.__activation = self.__activation * bias


            approx = np.dot(self.__dictionary,self.__activation)
            w = self.__data/approx
            w[np.isnan(w)] = 0
            wh = np.dot(w,np.transpose(self.__activation))

            wt = np.ones([self.__k,1],np.float32)
            wt = sum(np.transpose(self.__activation[:]))
            wt = np.transpose(wt)

            bias = wh/wt
            self.__dictionary = self.__dictionary *bias
            counter += 1
        return self.__dictionary,self.__activation


    def separate_is_without_template(self,iter=200):
        """
        テンプレートなしのIS仕様の分離処理を行う
        :param iter:
        :return:
        """
        counter = 0

        while counter < iter:
            approx = np.dot(self.__dictionary , self.__activation)
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

            self.__activation = self.__activation * bias

            approx = np.dot(self.__dictionary , self.__activation)
            w1 = self.__data/approx
            w2 = self.__activation/sum(approx[:])
            w1[np.isnan(w1)] = 0
            w2[np.isnan(w2)] = 0

            wh = np.dot(w1,np.transpose(w2))
            wt = sum(np.transpose(w2[:]))

            bias = wh/wt
            bias[np.isnan(bias)] = 0
            self.__dictionary = self.__dictionary * bias



            counter += 1

        return self.__dictionary,self.__activation
