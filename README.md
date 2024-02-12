## ディープラーニングの学習

   をしているつもりで、おれは Chainer のチュートリアルをやってたんだ。
   
   https://tutorials.chainer.org/ja/tutorial.html

   いや、間違ってないけどな。
   
   コンテナは GPU を使うために、compassionate_joliot cupy/cupy

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

## 0. PyTorch入門(Learn the Basics)

### [1] テンソル（Tensors） 10_pytorch-tensorsl.py
        まあなんとなくはテンソルも行列だよね。
        それもgpuで処理可能と

### [2] データセットとデータローダー（Datasets & DataLoaders） 11_pytorch_dataset.py
        PyTorchのデータセットは何となく解った。
        data（テンソル）と、ラベル（int??）の２つの要素を持つタプルを返すと。
        DataLoaders は、まだよくわからんけど、後から出てくるだろう。

### [3] データ変換（Transforms） 11_pytorch_dataset.py
        さっぱり解らんけど、また必要になったら戻ってくるか。

### [4] モデル構築（Build Model） 12_pytorch_model.py
        ここはまじめにじっくりやるか。
        じっくりやっても、解らんもんは解らんなぁ・・・・
        あとで、もう一度戻ってこんなんなぁ・・・・

### [5] 自動微分（Autograd）13_pytorch_autogrand.py
        うーん、、、[3] データ構造あたりから、さっぱり解らん・・・・・・・
        こっちのルートを通ったけど、一旦中断してチュートリアルルートに行くか・・・・

### [6] 最適化（Optimization Loop）
### [7] モデルの保存・読み込み（Save, Load and Use Model）
### [8] クイックスタート（Quickstart）
        一旦パス

## 1. PyTorch基礎(Learning PyTorch)

### [1] PyTorch60分講座: PyTorchとは？（DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ：WHAT IS PYTORCH?）
        なんか入門と変わらんな。
        NumpyベースでGPUをサポート
        Tensors（テンソル：多次元配列）

### [2] PyTorch60分講座: Autograd（自動微分）（AUTOGRAD: AUTOMATIC DIFFERENTIATION）
        21_autograd.py
        ザクっというと、x をインプットとした一連の計算結果をもとに、
        その計算の傾きを勾配ベクトルで求めるってこったな。
        コーディング上のポイントは、tensor生成時に requires_grad=True を指定する。
        計算後にout.backward()でgradを計算し、x.gradで勾配ベクトルを取得する。

### [3] PyTorch60分講座: ニューラルネットワーク入門（NEURAL NETWORKS）        
        22_nn.py
        なんか、nn.Conv2dとはなんぞや？
        nn.Linearとはなんぞや？
        勾配ベクトルを計算できる出力ノードとはなんぞや？
        って感じで止まってしまうなぁ・・・・・

        一旦あきらめて、
        https://euske.github.io/introdl/index.html
        をやってみるか。


