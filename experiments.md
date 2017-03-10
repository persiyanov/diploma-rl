## BePolite experiment

* Finetune both encoder & decoder

| | avg-reward | perplexity
--- | --- | ---
__Baseline seq2seq__ | -0.238 | 3.216

```
```

We finetune baseline with A2C using the following loss: `-llh * alpha + a2c_loss`, where alpha is scalar. The greater alpha is chosen, the more conservative model will be. All experiments were ran three times (for __500 batches__ with 64 samples each) in order to study stability of different setups (because sometimes llh-loss can blow up). `-` in table means that model blowed up. 

| A2C finetuned seq2seq | | |
--- | --- | ---
| __llh-alpha__ | __avg-reward__ | __perplexity__
5 | `-`, `-`, `-`  | `-`, `-`, `-`
10 | `-`, `-`, -0.065| `-`, `-`, 3.483
20 | -0.061, `-`, `-`  | 3.066, `-`, `-`
50 | -0.102, -0.073, `-` | 3.073, 3.081, `-`
70 | ... | ...
100 | ... | ...
