## おれは Chainer のチュートリアルをやってたんだ。
https://tutorials.chainer.org/ja/tutorial.html
コンテナはrtx3060ti を使うために、compassionate_joliot cupy/cupy

### 8. NumPy 入門        01_nump.py
        多次元配列（multidimensional array）を扱うライブラリだと
        例えば、
        様々な機械学習手法を統一的なインターフェースで利用できる scikit-learn や、
        ニューラルネットワークの記述・学習を行うためのフレームワークである Chainer は、
        NumPy に慣れておくことでとても使いやすくなります。
        ですと。

### 9. scikit-learn 入門 02_scikit-sample.py
        Python のオープンソース機械学習ライブラリ

### 10. CuPy 入門        03_matrix_calc.py
        NumPy と高い互換性を持つ数値計算ライブラリで
        NVIDIA GPU で実行することで簡単に高速化できる
        構築がややこしそうだったから、cupy コンテナを使う事にした。
            docker pull cupy/cupy:latest

### 11. Pandas 入門      04_panda-sam.py
        データ操作ライブラリ
        CSVのR/Wや、並べ替えとか欠損値の除去 / 補間とかができる

### 12. Matplotlib 入門  05_plot-sam.py 
        グラフの描画
        コンテナで使うと（guiが無いから）グラフが表示できない。
        Xサーバー使うとかもあるみたいだけど、めんどくさいからpngに出力する方向で利用

### 14. Chainer の基礎   06_chain.py
        ディープラーニングフレームワーク
        もう開発が終了していて、PyTorchに移ったみたい。（＝PyTorchも見ないと）
        要は
            scikit-learn：マシンラーニング（機械学習）
            chainer：ディープラーニング（深層学習）
        ってことやね。

## tensorflowも開発環境だけは作っておいて。
https://github.com/sasaki2110/tensorflow001
コンテナは　intelligent_curie
これは今しばらくは塩漬けにしておく

## で、PyTorchチュートリアル（日本語翻訳版）を始める
https://yutaroogawa.github.io/pytorch_tutorials_jp/
コンテナは、compassionate_joliot cupy/cupyに戻る。

### [1] テンソル（Tensors） 10_pytorch-tensorsl.py
        まあなんとなくはテンソルも行列だよね。
        それもgpuで処理可能と

### [2] データセットとデータローダー（Datasets & DataLoaders） 11_pytorch_dataset.py
        PyTorchのデータセットは何となく解った。
        data（テンソル）と、ラベル（int??）の２つの要素を持つタプルを返すと。
        DataLoaders は、まだよくわからんけど、後から出てくるだろう。

### [3] データ変換（Transforms） 今からこれ