# -*- coding: utf-8 -*-

import numpy as np

from NMF import NMF


"""
テスト用のスクリプト
使い方記載
"""
if __name__ == "__main__":

    nmf = NMF()

    """
    コンストラクタを作ったあとには、まずこれを呼ぶこと
    ここで、k,row,columnの設定をするので、これを呼ばないとsetDictionaryが動かない
    """
    nmf.setAnalyzData(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5]
        ],
        k=3
    )

    """
    ここでテンプレートをセットする
    """
    nmf.setDictionary(0, [0.0, 2.0])
    nmf.setDictionary(1, [1.0, 6.0])
    nmf.setDictionary(2, [11.0, 10.0])

    """
    NMF開始
    引数には、アルゴリズムと反復更新回数を渡しておく
    """
    dic, act = nmf.separate_euc_with_template(iter=200)
    # dic,act = nmf.separate_kl_with_template(iter=200)
    # dic,act = nmf.separate_is_with_template(iter=200)

    # dic,act = nmf.separate_euc_without_template(iter=200)
    # dic,act = nmf.separate_kl_without_template(iter=200)
    # dic,act = nmf.separate_is_without_template(iter=200)
    """
    結果表示
    """
    print("=========================== Dictionary ===========================")
    print(dic)
    print("=========================== Activation ===========================")
    print(act)

    """
    ちゃんと分解できているかの確認
    """
    print("=========================== Approx ===========================")
    print(np.dot(dic, act))
