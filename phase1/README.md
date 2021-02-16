# Phase1
Phase1ではTemporal pairと呼ばれる同様な撮像条件で同じ場所を異なる時期に撮影した画像のペアを作り，そこから新しいクレータを検出する．


# Usage
## detect_new_crater.pyの実行
プログラム中でNAC画像ダウンロードから検出までしている．プログラム実行の際にはcsvファイル2つを入力する必要がある．
cratert_list.csvは調べる地点，done_list.csvは調べ終わったペアが記されており，途中で終了してもこのcsvがあることによって途中から再開することができる．

```bash
python ../csv/detect_new_crater.py ../csv/crater_list.csv done_list.csv
```

LROCのHPはよく落ちるため，途中でプログラムが止まっていることがあるのに注意する．差分があったものはconst.OUTPUT_PATH下に出力される．


## クレーターのチェック
const.OUTPUT_PATH下に出力されたものが本当にクレーターなのか目視でチェックする．クレーターだったものはナンバリングしながらconst.NEW_CRATERS_PATH下に設置する．
また，crater_list.csvをconst.const.CSV_PATH下に作成し，検出したクレーターのIDとピクセル数を記入する．


## 画像の変換
プログラム実行直後の画像はpngなのでmulti-tiffにして変化を見やすくする．

```bash
python convert_png_to_tiff.py
```

必要に応じてコメントアウトをしたり，42/53/63行目を変化させる．


## データの正規化
DBの第二正規系くらいまでデータをまとめる．

```bash
python summarize_data.py
```


# Note
## ペアを作る際の条件
module下のdownload.pyのmake_temporal_pairを変更すると，Temporal pairを作る際の条件が変更できる．
