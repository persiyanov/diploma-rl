## BePolite experiment

* Finetune both encoder & decoder

| | avg-reward | perplexity
--- | --- | ---
__Baseline seq2seq__ | -0.238 | 3.216

```
```

We finetune baseline with A2C using the following loss: `-llh * alpha + a2c_loss`, where alpha is scalar. The greater alpha is chosen, the more conservative model will be. All experiments were ran several times (for __500 batches__ with 64 samples each) in order to study stability of different setups (because sometimes llh-loss can blow up). Values for different runs are separated by commas. `-` in table means that model blowed up. 

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
