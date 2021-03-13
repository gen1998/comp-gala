# comp-gala
成績はふるいませんでしたが反省を込めて簡単にまとめておきます。
## Data Augmentation
- 村上が教えてくれたalbumentationを使ってみました。PadIfNeededを用いて(235,80)の画像にして計算を行いました。
- 初めはただ単に、(80,80)のサイズでリサイズして学習を行っていましたが、pandIfNeededを用いた方が精度はよくなりました。
- あとは反転して、彩度と明度を変換した画像も生成しました。対して精度は高くならなかった印象です。
- keras.preprocessing.image.ImageDataGeneratorに元々あるズーム機能や、回転機能も用いていました。
## Model
- vgg16, resnet50, xceptionを用いてfine-tuningを行いました。
- efficientnetは最終日使ってみたのですが、時間的にうまく調整できませんでした。
## 実行方法
- 直下にあるrun.pyで実行
  - model.image_create()でImageDataGeneratorで使用するフォルダーを作成し、学習に適用。
  - model.train()で学習を行う。
## 反省
- 転移学習を用いる際、層を凍結して学習は行いませんでした。そこに7割と8割の差があったと考えています。
- 画像のresizeは素直に正方形の大きいサイズの方がうまくいったかもしれません。(80,80)に留めてしまい、それ以上の大きさの学習は行いませんでした。
