# Examples on NMT

In this example, we test AdaMod on the IWSLT'14 De-En and WMT'14 En-De datasets, comparing with Adam. The implementation is highly based on [fairseq repo](https://github.com/pytorch/fairseq/tree/master/examples/translation).

Tested with PyTorch 1.1.0.

## Settings

### IWSLT'14 German to English (Transformer-small)
After downloading and preprocessing the data (please refer to the original [ repo](https://github.com/pytorch/fairseq/tree/master/examples/translation)), we'll train a Transformer-small model over this data:
```bash
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en \
    --optimizer adamod --adam-betas '(0.9, 0.98)' --beta3 0.9999 \
    --lr 5e-4 --lr-scheduler cold_start --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4000 --max-update 50000
```
Note that for fair comparison with Adam, we still hold `--warmup-updates 4000` for this setting. In fact, on the IWSLT'14 De-En dataset, AdaMod does not depend on any `--lr-scheduler`. A constant lr (e.g. 5e-4) can achieve higher BLEU4 score up to `35.1`. What's more, if you further use `--update-freq` option for delay updating, the state-of-the-art result `35.6` will be achieved.

Then you need to average 10 latest checkpoints:
```bash
python scripts/average_checkpoints.py --inputs checkpoints/transformer \
   --num-epoch-checkpoints 10 --output checkpoints/transformer/model.pt
```

Finally you can evaluate trained model:
```bash
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/transformer/model.pt \
    --batch-size 128 --beam 5 --remove-bpe
```

### WMT'14 English to German (Transformer-base/big)
Similarly, after processing the data (following [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)), you can train a new model on this data.

For Transformer-base:
```bash
fairseq-train data-bin/wmt14_en_de \
  --arch transformer_wmt_en_de \
  --optimizer adamod --adam-betas '(0.9, 0.98)' --beta3 0.999 --clip-norm 0.0 \
  --lr-scheduler cold_start --warmup-updates 4000 \
  --lr 0.0005 --min-lr 1e-09 \
  --dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 3584 --max-epoch 50 \
  --fp16
```
Note that the --fp16 flag requires you have CUDA 9.1 or greater and a Volta GPU.

For Transformer-big:
```bash
fairseq-train data-bin/wmt14_en_de \
  --arch transformer_vaswani_wmt_en_de_big \
  --optimizer adamod --adam-betas '(0.9, 0.98)' --beta3 0.9999 --clip-norm 0.0 \
  --lr-scheduler cold_start --warmup-updates 4000 \
  --lr 0.0005 --min-lr 1e-09 \
  --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 3584 --max-epoch 50 \
  --fp16
```

Then you need to average 10 latest checkpoints:
```bash
python scripts/average_checkpoints.py --inputs checkpoints/transformer \
   --num-epoch-checkpoints 10 --output checkpoints/transformer/model.pt
```

Finally you can evaluate trained model:
```bash
python generate.py data-bin/wmt14_en_de \
  --path checkpoints/transformer/model.pt \
  --batch-size 128 --beam 4 --remove-bpe --lenpen 0.6
```
