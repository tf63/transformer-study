# Transformer勉強会 実装

環境構築
```
    bash docker.sh build
    bash docker.sh shell
```

学習
```
    bash cmd/train.sh
```

推論
```
    python3 inference.py
```

### 

### 参考リンク

model.pyの参考
- https://qiita.com/gensal/items/e1c4a34dbfd0d7449099

torch.nn.module.transformer の内部実装
- https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py

pytorch lightning公式
- https://pytorch-lightning.readthedocs.io/en/2.2.3/common/lightning_module.html

デモ用プロジェクト
- https://github.com/i14kwmr/practice-transformer/tree/main
