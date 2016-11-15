#coding:utf-8

import numpy as np

"""
面倒くさいけれど、この２つをimportしないといけない
"""
from NMF import NMF
import Update


"""
テスト用のスクリプト
使い方記載
"""
if __name__ == "__main__":

    """
    基本的にはコンストラクタで面倒なk,row,columnを渡さなくてもOK
    適当に作っただけなので・・・
    """
    nmf = NMF()

    """
    コンストラクタを作ったあとには、まずこれを呼んでください！！！！！
    ここで、k,row,columnの設定をするので、これを呼ばないとsetDictionaryが動かなくなります！！！！！！
    """
    nmf.setAnalyzData([[1,2,3,4],[2,3,4,5]],k=3)

    """
    ここでテンプレートをセットする
    """
    nmf.setDictionary(0,[0.0,2.0])
    nmf.setDictionary(1,[1.0,6.0])
    nmf.setDictionary(2,[11.0,10.0])

    """
    NMF開始
    引数には、アルゴリズムと反復更新回数を渡しておく
    """
    dic,act = nmf.start(algf=Update.i_s,iter=200000)

    """
    結果表示（dicは返す必要ない）
    """
    print dic
    print act

    """
    ちゃんと分解できているかの確認
    """
    print np.dot(dic,act)
