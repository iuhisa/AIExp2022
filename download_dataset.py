'''
1.datasetをダウンロードする
2.datasetに含まれる画像のPathを列挙する(Pathリスト)
3.Pathリストを train, val, test に分けて、train.txt, val.txt, test.txt などに保存する(割合は指定する)
(4. datasetをロードするときはこのPathリストを見て、ロードする。)

以下は、ドメイン分けされたunpairなデータセットの例
pairのあるデータセットだったり、タスクによって構成が変わり得る

./ ┬ datasets ┬ flower_pansy ┬ images ──┬ 001.png
                             │          ├ 002.png
                             ├ train.txt
                             ├ val.txt 要る???
                             └ test.txt
              ├ flower_dandelion ┬ ...

'''