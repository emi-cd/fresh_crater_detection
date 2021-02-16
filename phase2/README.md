# Phase2
Phase1で集めた新しいクレーターで学習させて，物体検出で新しいクレーターが検出する．ここではYOLOv4を用いる．

# Usage
## YOLOv4の準備
[YOLOv4のコード](https://github.com/AlexeyAB/darknet)をクローンする．2021年2月現在も開発が進んでいるため，基本的にAlexeyさんが書いてある通りに進めてほしい．
参考までに本研究で用いたcfgファイルなどをtask下に置く．


## データの下準備
Phase1で集めた新しいクレーターの画像はbefore画像に合わせて射影変換しているため，元のNAC画像から変換前の画像を抜き出してくる必要がある．学習用データはconst.CRATER_TRAIN_DATA下に出力される．

```bash
python make_data_for_ml.py
```

## アノテーション
私は[labelImg](https://github.com/tzutalin/labelImg)を使用したがお好みで．


## データの水増しとデータの分割
const.CRATER_TRAIN_DATA下のデータに対してtrain data, valid data, test dataに分割し，ランダムに水増しする．実行前にYOLOv4コードとのディレクトリ関係を確認する．
```bash
python process.py
```
分割する割合を変化させる場合は153,154行目を変化させる．


## 学習
AlexeyさんのREADME参照
