# Transformer勉強会 実装
解説記事
- https://qiita.com/tf63/items/788eeecd458acfa78c83

### 動作確認

**.envの作成**
- [Weights & Biases](https://www.wandb.jp/) のアカウントを作成する
- サインインし **User Setting > Danger zone > API keys** からAPIキーを取得する
- `.env.example`をコピーし`.env`という名前のファイルを作成する
- `.env`に取得したAPIキーを書く (**APIキーは公開しない．.envはgit管理から外しているので大丈夫**)
```
    WANDB_API_KEY="<API Key>"
```

**環境構築**
```
    bash cmd/docker.sh build
    bash cmd/docker.sh shell
```

**学習**
```
    bash cmd/train.sh
```

あるいは
```
    python3 train.py \
        --accelerator gpu \
        --devices 1 \
        --batch_size 256 \
        --num_datas 50000 \
        --max_epochs 10 \
        --lr 0.0001 \
        --num_heads 8 \
        --dim 512
```

もしかしたら初回は`wandb login`する必要があるかもしれない

**推論**
```
    python3 inference.py
```


### dependency
```
torch==2.0.0
pytorch-lightning==2.2.3
wandb==0.16.6
click==8.1.7
jupyter==1.0.0
ipykernel==6.29.4 
```

### 参考リンク

torch.nn.module.transformer の内部実装
- https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py

pytorch lightning公式
- https://pytorch-lightning.readthedocs.io/en/2.2.3/common/lightning_module.html

huggingface/transformers
- https://github.com/huggingface/transformers

model.pyの参考
- https://qiita.com/gensal/items/e1c4a34dbfd0d7449099


デモ用プロジェクトの参考
- https://github.com/i14kwmr/practice-transformer/tree/main
