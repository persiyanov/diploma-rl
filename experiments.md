# BePolite experiment

* Finetune both encoder & decoder

| | avg-reward | perplexity
--- | --- | ---
__Baseline seq2seq__ | -0.238 | 3.216

```
```

We finetune baseline with A2C using the following loss: `-llh * alpha + a2c_loss`, where alpha is scalar. The greater alpha is chosen, the more conservative model will be. All experiments were ran several times (for __500 batches__ with 64 samples each) in order to study stability of different setups (because sometimes llh-loss can blow up). Values for different runs are separated by commas. `-` in table means that model blowed up.

__Critic architecture (input is an lstm hidden state)__: Input(1024)->Dense(2048, ReLU)->Dense(1024, ReLU)->Dense(512, ReLU)->Dense(1, linear)

| A2C finetuned seq2seq | | |
--- | --- | ---
| __llh-alpha__ | __avg-reward__ | __perplexity__
5 | `-`, `-`, `-`  | `-`, `-`, `-`
10 | `-`, `-`, -0.065| `-`, `-`, 3.483
20 | -0.059, `-`, `-`  | 3.242, `-`, `-`
50 | -0.06, -0.05, -0.06 | 3.225, 3.225, 3.226
70 | -0.08, -0.07, -0.08 | 3.225, 3.224, 3.225
100 | -0.07, -0.08, -0.07| 3.225, 3.224, 3.225

```
```
* Gradually reduce alpha in 500 batches [90,70,50,30,20]

| avg-reward | perplexity |
--- | ---
-0.04, `-`, -0.05 | 3.188, `-`, 3.201

```
```
--------
### New seq2seq baseline on the same data, but early stopped using validation set.
| | avg-reward | perplexity
--- | --- | ---
__Baseline seq2seq__ | -0.136 | 3.142


__Critic architecture (input is an lstm hidden state)__: Input(1024)->Dense(512, ReLU)->Dense(512, ReLU)->Dense(256, ReLU)->Dense(1, linear)

### Finetuned seq2seq with A2C
| A2C finetuned seq2seq | | |
--- | --- | ---
| __llh-alpha__ | __avg-reward__ | __perplexity__
5 | -0.021  | 3.297
20 | -0.065 | 3.270

```
```

------
# BeLikeX experiment

Finetune on user `24203097`

| Model name | Perplexity | Perplexity/uid | Avg. Reward |
--- | --- | --- | ---
baseline LM | 4.235 | 5.249 | 0.258
llh on user | 5.792  | 6.540 | 0.389
dssm weighting | 4.337 | 5.358 | 0.281
scst on dssm, alpha=0.5 | 4.760 | 5.830 | 0.294

