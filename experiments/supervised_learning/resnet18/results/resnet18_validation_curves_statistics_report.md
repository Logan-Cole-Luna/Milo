# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 3

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 2.0661 | 0.0133 | 0.0077 | 2.0330 | 2.0992 |
| MILO_LW | 2.3276 | 0.0060 | 0.0035 | 2.3127 | 2.3425 |
| MUON | 2.6896 | 0.0478 | 0.0276 | 2.5710 | 2.8083 |
| ADALAYER | 3.0748 | 0.3911 | 0.2258 | 2.1032 | 4.0465 |
| ADAM_MINI | 2.7253 | 0.0133 | 0.0077 | 2.6923 | 2.7584 |
| SGD | 2.4103 | 0.0477 | 0.0275 | 2.2919 | 2.5288 |
| ADAMW | 2.9377 | 0.2664 | 0.1538 | 2.2760 | 3.5994 |
| ADAGRAD | 2.3245 | 0.1366 | 0.0788 | 1.9852 | 2.6637 |
| NOVOGRAD | 2.5619 | 0.1190 | 0.0687 | 2.2663 | 2.8575 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better    |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:----------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  2.06611 |  2.32761 | MILO      | 0.000130429 | ***           | final_validation_loss |
| MILO          | MUON          |  2.06611 |  2.68963 | MILO      | 0.00100489  | **            | final_validation_loss |
| MILO          | ADALAYER      |  2.06611 |  3.07484 | MILO      | 0.0464907   | *             | final_validation_loss |
| MILO          | ADAM_MINI     |  2.06611 |  2.72533 | MILO      | 4.42493e-07 | ***           | final_validation_loss |
| MILO          | SGD           |  2.06611 |  2.41034 | MILO      | 0.00390834  | **            | final_validation_loss |
| MILO          | ADAMW         |  2.06611 |  2.93769 | MILO      | 0.0294864   | *             | final_validation_loss |
| MILO          | ADAGRAD       |  2.06611 |  2.32445 | MILO      | 0.0805141   |               | final_validation_loss |
| MILO          | NOVOGRAD      |  2.06611 |  2.56192 | MILO      | 0.0176684   | *             | final_validation_loss |
| MILO_LW       | MUON          |  2.32761 |  2.68963 | MILO_LW   | 0.0051789   | **            | final_validation_loss |
| MILO_LW       | ADALAYER      |  2.32761 |  3.07484 | MILO_LW   | 0.0804301   |               | final_validation_loss |
| MILO_LW       | ADAM_MINI     |  2.32761 |  2.72533 | MILO_LW   | 4.04374e-05 | ***           | final_validation_loss |
| MILO_LW       | SGD           |  2.32761 |  2.41034 | MILO_LW   | 0.0930178   |               | final_validation_loss |
| MILO_LW       | ADAMW         |  2.32761 |  2.93769 | MILO_LW   | 0.0579927   |               | final_validation_loss |
| MILO_LW       | ADAGRAD       |  2.32761 |  2.32445 | ADAGRAD   | 0.971657    |               | final_validation_loss |
| MILO_LW       | NOVOGRAD      |  2.32761 |  2.56192 | MILO_LW   | 0.0759172   |               | final_validation_loss |
| MUON          | ADALAYER      |  2.68963 |  3.07484 | MUON      | 0.228877    |               | final_validation_loss |
| MUON          | ADAM_MINI     |  2.68963 |  2.72533 | MUON      | 0.323817    |               | final_validation_loss |
| MUON          | SGD           |  2.68963 |  2.41034 | SGD       | 0.00200634  | **            | final_validation_loss |
| MUON          | ADAMW         |  2.68963 |  2.93769 | MUON      | 0.24585     |               | final_validation_loss |
| MUON          | ADAGRAD       |  2.68963 |  2.32445 | ADAGRAD   | 0.0323845   | *             | final_validation_loss |
| MUON          | NOVOGRAD      |  2.68963 |  2.56192 | NOVOGRAD  | 0.195785    |               | final_validation_loss |
| ADALAYER      | ADAM_MINI     |  3.07484 |  2.72533 | ADAM_MINI | 0.261692    |               | final_validation_loss |
| ADALAYER      | SGD           |  3.07484 |  2.41034 | SGD       | 0.0965762   |               | final_validation_loss |
| ADALAYER      | ADAMW         |  3.07484 |  2.93769 | ADAMW     | 0.645382    |               | final_validation_loss |
| ADALAYER      | ADAGRAD       |  3.07484 |  2.32445 | ADAGRAD   | 0.0668041   |               | final_validation_loss |
| ADALAYER      | NOVOGRAD      |  3.07484 |  2.56192 | NOVOGRAD  | 0.141835    |               | final_validation_loss |
| ADAM_MINI     | SGD           |  2.72533 |  2.41034 | SGD       | 0.00479009  | **            | final_validation_loss |
| ADAM_MINI     | ADAMW         |  2.72533 |  2.93769 | ADAM_MINI | 0.301253    |               | final_validation_loss |
| ADAM_MINI     | ADAGRAD       |  2.72533 |  2.32445 | ADAGRAD   | 0.0354993   | *             | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      |  2.72533 |  2.56192 | NOVOGRAD  | 0.138803    |               | final_validation_loss |
| SGD           | ADAMW         |  2.41034 |  2.93769 | SGD       | 0.0712426   |               | final_validation_loss |
| SGD           | ADAGRAD       |  2.41034 |  2.32445 | ADAGRAD   | 0.393453    |               | final_validation_loss |
| SGD           | NOVOGRAD      |  2.41034 |  2.56192 | SGD       | 0.145902    |               | final_validation_loss |
| ADAMW         | ADAGRAD       |  2.93769 |  2.32445 | ADAGRAD   | 0.0384697   | *             | final_validation_loss |
| ADAMW         | NOVOGRAD      |  2.93769 |  2.56192 | NOVOGRAD  | 0.119341    |               | final_validation_loss |
| ADAGRAD       | NOVOGRAD      |  2.32445 |  2.56192 | ADAGRAD   | 0.0869188   |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 3

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 47.3877 | 0.5083 | 0.2935 | 46.1249 | 48.6504 |
| MILO_LW | 40.7111 | 0.6698 | 0.3867 | 39.0473 | 42.3750 |
| MUON | 50.4840 | 0.4672 | 0.2697 | 49.3235 | 51.6444 |
| ADALAYER | 39.9358 | 3.6877 | 2.1291 | 30.7751 | 49.0965 |
| ADAM_MINI | 40.6272 | 0.7213 | 0.4164 | 38.8354 | 42.4189 |
| SGD | 43.3679 | 0.9259 | 0.5346 | 41.0677 | 45.6681 |
| ADAMW | 40.4494 | 4.1932 | 2.4209 | 30.0330 | 50.8657 |
| ADAGRAD | 42.1877 | 2.0780 | 1.1997 | 37.0256 | 47.3497 |
| NOVOGRAD | 39.1160 | 1.4000 | 0.8083 | 35.6382 | 42.5939 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better    |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:----------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  47.3877 |  40.7111 | MILO      | 0.00024788  | ***           | final_validation_accuracy |
| MILO          | MUON          |  47.3877 |  50.484  | MUON      | 0.00152385  | **            | final_validation_accuracy |
| MILO          | ADALAYER      |  47.3877 |  39.9358 | MILO      | 0.070214    |               | final_validation_accuracy |
| MILO          | ADAM_MINI     |  47.3877 |  40.6272 | MILO      | 0.00035025  | ***           | final_validation_accuracy |
| MILO          | SGD           |  47.3877 |  43.3679 | MILO      | 0.00638148  | **            | final_validation_accuracy |
| MILO          | ADAMW         |  47.3877 |  40.4494 | MILO      | 0.101152    |               | final_validation_accuracy |
| MILO          | ADAGRAD       |  47.3877 |  42.1877 | MILO      | 0.0426224   | *             | final_validation_accuracy |
| MILO          | NOVOGRAD      |  47.3877 |  39.116  | MILO      | 0.00475364  | **            | final_validation_accuracy |
| MILO_LW       | MUON          |  40.7111 |  50.484  | MUON      | 7.47468e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADALAYER      |  40.7111 |  39.9358 | MILO_LW   | 0.752521    |               | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  40.7111 |  40.6272 | MILO_LW   | 0.889744    |               | final_validation_accuracy |
| MILO_LW       | SGD           |  40.7111 |  43.3679 | SGD       | 0.0189689   | *             | final_validation_accuracy |
| MILO_LW       | ADAMW         |  40.7111 |  40.4494 | MILO_LW   | 0.924303    |               | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  40.7111 |  42.1877 | ADAGRAD   | 0.344136    |               | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  40.7111 |  39.116  | MILO_LW   | 0.177239    |               | final_validation_accuracy |
| MUON          | ADALAYER      |  50.484  |  39.9358 | MUON      | 0.0365817   | *             | final_validation_accuracy |
| MUON          | ADAM_MINI     |  50.484  |  40.6272 | MUON      | 0.000116402 | ***           | final_validation_accuracy |
| MUON          | SGD           |  50.484  |  43.3679 | MUON      | 0.00137336  | **            | final_validation_accuracy |
| MUON          | ADAMW         |  50.484  |  40.4494 | MUON      | 0.051959    |               | final_validation_accuracy |
| MUON          | ADAGRAD       |  50.484  |  42.1877 | MUON      | 0.0164955   | *             | final_validation_accuracy |
| MUON          | NOVOGRAD      |  50.484  |  39.116  | MUON      | 0.00243471  | **            | final_validation_accuracy |
| ADALAYER      | ADAM_MINI     |  39.9358 |  40.6272 | ADAM_MINI | 0.778256    |               | final_validation_accuracy |
| ADALAYER      | SGD           |  39.9358 |  43.3679 | SGD       | 0.244641    |               | final_validation_accuracy |
| ADALAYER      | ADAMW         |  39.9358 |  40.4494 | ADAMW     | 0.881271    |               | final_validation_accuracy |
| ADALAYER      | ADAGRAD       |  39.9358 |  42.1877 | ADAGRAD   | 0.421756    |               | final_validation_accuracy |
| ADALAYER      | NOVOGRAD      |  39.9358 |  39.116  | ADALAYER  | 0.74643     |               | final_validation_accuracy |
| ADAM_MINI     | SGD           |  40.6272 |  43.3679 | SGD       | 0.0174532   | *             | final_validation_accuracy |
| ADAM_MINI     | ADAMW         |  40.6272 |  40.4494 | ADAM_MINI | 0.948565    |               | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  40.6272 |  42.1877 | ADAGRAD   | 0.3231      |               | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  40.6272 |  39.116  | ADAM_MINI | 0.195364    |               | final_validation_accuracy |
| SGD           | ADAMW         |  43.3679 |  40.4494 | SGD       | 0.350986    |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  43.3679 |  42.1877 | SGD       | 0.440188    |               | final_validation_accuracy |
| SGD           | NOVOGRAD      |  43.3679 |  39.116  | SGD       | 0.0161345   | *             | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  40.4494 |  42.1877 | ADAGRAD   | 0.566847    |               | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  40.4494 |  39.116  | ADAMW     | 0.645015    |               | final_validation_accuracy |
| ADAGRAD       | NOVOGRAD      |  42.1877 |  39.116  | ADAGRAD   | 0.110675    |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 3

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.4719 | 0.0050 | 0.0029 | 0.4595 | 0.4843 |
| MILO_LW | 0.4028 | 0.0079 | 0.0046 | 0.3832 | 0.4225 |
| MUON | 0.5006 | 0.0001 | 0.0001 | 0.5003 | 0.5009 |
| ADALAYER | 0.4008 | 0.0350 | 0.0202 | 0.3139 | 0.4878 |
| ADAM_MINI | 0.4048 | 0.0034 | 0.0020 | 0.3964 | 0.4133 |
| SGD | 0.4317 | 0.0082 | 0.0047 | 0.4115 | 0.4520 |
| ADAMW | 0.4009 | 0.0439 | 0.0253 | 0.2919 | 0.5099 |
| ADAGRAD | 0.4228 | 0.0198 | 0.0114 | 0.3736 | 0.4719 |
| NOVOGRAD | 0.3899 | 0.0079 | 0.0046 | 0.3703 | 0.4096 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better    |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:----------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.471912 | 0.402843 | MILO      | 0.000566072 | ***           | final_validation_f1_score |
| MILO          | MUON          | 0.471912 | 0.500577 | MUON      | 0.00988246  | **            | final_validation_f1_score |
| MILO          | ADALAYER      | 0.471912 | 0.400834 | MILO      | 0.0694042   |               | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.471912 | 0.404815 | MILO      | 0.000105148 | ***           | final_validation_f1_score |
| MILO          | SGD           | 0.471912 | 0.43171  | MILO      | 0.00379774  | **            | final_validation_f1_score |
| MILO          | ADAMW         | 0.471912 | 0.400897 | MILO      | 0.105389    |               | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.471912 | 0.422754 | MILO      | 0.0429454   | *             | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.471912 | 0.389906 | MILO      | 0.00031908  | ***           | final_validation_f1_score |
| MILO_LW       | MUON          | 0.402843 | 0.500577 | MUON      | 0.00217366  | **            | final_validation_f1_score |
| MILO_LW       | ADALAYER      | 0.402843 | 0.400834 | MILO_LW   | 0.930867    |               | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.402843 | 0.404815 | ADAM_MINI | 0.720888    |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.402843 | 0.43171  | SGD       | 0.011718    | *             | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.402843 | 0.400897 | MILO_LW   | 0.946253    |               | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.402843 | 0.422754 | ADAGRAD   | 0.216877    |               | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.402843 | 0.389906 | MILO_LW   | 0.115785    |               | final_validation_f1_score |
| MUON          | ADALAYER      | 0.500577 | 0.400834 | MUON      | 0.0386738   | *             | final_validation_f1_score |
| MUON          | ADAM_MINI     | 0.500577 | 0.404815 | MUON      | 0.000413094 | ***           | final_validation_f1_score |
| MUON          | SGD           | 0.500577 | 0.43171  | MUON      | 0.00463317  | **            | final_validation_f1_score |
| MUON          | ADAMW         | 0.500577 | 0.400897 | MUON      | 0.0589588   |               | final_validation_f1_score |
| MUON          | ADAGRAD       | 0.500577 | 0.422754 | MUON      | 0.0208829   | *             | final_validation_f1_score |
| MUON          | NOVOGRAD      | 0.500577 | 0.389906 | MUON      | 0.00169457  | **            | final_validation_f1_score |
| ADALAYER      | ADAM_MINI     | 0.400834 | 0.404815 | ADAM_MINI | 0.862385    |               | final_validation_f1_score |
| ADALAYER      | SGD           | 0.400834 | 0.43171  | SGD       | 0.263362    |               | final_validation_f1_score |
| ADALAYER      | ADAMW         | 0.400834 | 0.400897 | ADAMW     | 0.998559    |               | final_validation_f1_score |
| ADALAYER      | ADAGRAD       | 0.400834 | 0.422754 | ADAGRAD   | 0.411467    |               | final_validation_f1_score |
| ADALAYER      | NOVOGRAD      | 0.400834 | 0.389906 | ADALAYER  | 0.646153    |               | final_validation_f1_score |
| ADAM_MINI     | SGD           | 0.404815 | 0.43171  | SGD       | 0.0176391   | *             | final_validation_f1_score |
| ADAM_MINI     | ADAMW         | 0.404815 | 0.400897 | ADAM_MINI | 0.89149     |               | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.404815 | 0.422754 | ADAGRAD   | 0.255091    |               | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.404815 | 0.389906 | ADAM_MINI | 0.0654888   |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.43171  | 0.400897 | SGD       | 0.347608    |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.43171  | 0.422754 | SGD       | 0.527168    |               | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.43171  | 0.389906 | SGD       | 0.00311917  | **            | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.400897 | 0.422754 | ADAGRAD   | 0.493146    |               | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.400897 | 0.389906 | ADAMW     | 0.708762    |               | final_validation_f1_score |
| ADAGRAD       | NOVOGRAD      | 0.422754 | 0.389906 | ADAGRAD   | 0.0873854   |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 3

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9610 | 0.0007 | 0.0004 | 0.9594 | 0.9626 |
| MILO_LW | 0.9546 | 0.0006 | 0.0003 | 0.9532 | 0.9560 |
| MUON | 0.9632 | 0.0009 | 0.0005 | 0.9610 | 0.9654 |
| ADALAYER | 0.9476 | 0.0078 | 0.0045 | 0.9282 | 0.9670 |
| ADAM_MINI | 0.9523 | 0.0005 | 0.0003 | 0.9512 | 0.9535 |
| SGD | 0.9530 | 0.0022 | 0.0013 | 0.9475 | 0.9584 |
| ADAMW | 0.9496 | 0.0086 | 0.0050 | 0.9283 | 0.9710 |
| ADAGRAD | 0.9551 | 0.0054 | 0.0031 | 0.9417 | 0.9684 |
| NOVOGRAD | 0.9498 | 0.0034 | 0.0020 | 0.9413 | 0.9583 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better    |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:----------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.960982 | 0.954588 | MILO      | 0.000239005 | ***           | final_validation_auc |
| MILO          | MUON          | 0.960982 | 0.963217 | MUON      | 0.0280871   | *             | final_validation_auc |
| MILO          | ADALAYER      | 0.960982 | 0.947576 | MILO      | 0.0957329   |               | final_validation_auc |
| MILO          | ADAM_MINI     | 0.960982 | 0.952349 | MILO      | 0.000101982 | ***           | final_validation_auc |
| MILO          | SGD           | 0.960982 | 0.952965 | MILO      | 0.017748    | *             | final_validation_auc |
| MILO          | ADAMW         | 0.960982 | 0.949646 | MILO      | 0.148529    |               | final_validation_auc |
| MILO          | ADAGRAD       | 0.960982 | 0.955074 | MILO      | 0.195735    |               | final_validation_auc |
| MILO          | NOVOGRAD      | 0.960982 | 0.949761 | MILO      | 0.0262225   | *             | final_validation_auc |
| MILO_LW       | MUON          | 0.954588 | 0.963217 | MUON      | 0.000365958 | ***           | final_validation_auc |
| MILO_LW       | ADALAYER      | 0.954588 | 0.947576 | MILO_LW   | 0.259327    |               | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.954588 | 0.952349 | MILO_LW   | 0.00708502  | **            | final_validation_auc |
| MILO_LW       | SGD           | 0.954588 | 0.952965 | MILO_LW   | 0.329528    |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.954588 | 0.949646 | MILO_LW   | 0.423491    |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.954588 | 0.955074 | ADAGRAD   | 0.890357    |               | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.954588 | 0.949761 | MILO_LW   | 0.131074    |               | final_validation_auc |
| MUON          | ADALAYER      | 0.963217 | 0.947576 | MUON      | 0.0720231   |               | final_validation_auc |
| MUON          | ADAM_MINI     | 0.963217 | 0.952349 | MUON      | 0.000322816 | ***           | final_validation_auc |
| MUON          | SGD           | 0.963217 | 0.952965 | MUON      | 0.00769381  | **            | final_validation_auc |
| MUON          | ADAMW         | 0.963217 | 0.949646 | MUON      | 0.109854    |               | final_validation_auc |
| MUON          | ADAGRAD       | 0.963217 | 0.955074 | MUON      | 0.116066    |               | final_validation_auc |
| MUON          | NOVOGRAD      | 0.963217 | 0.949761 | MUON      | 0.0160533   | *             | final_validation_auc |
| ADALAYER      | ADAM_MINI     | 0.947576 | 0.952349 | ADAM_MINI | 0.400252    |               | final_validation_auc |
| ADALAYER      | SGD           | 0.947576 | 0.952965 | SGD       | 0.354408    |               | final_validation_auc |
| ADALAYER      | ADAMW         | 0.947576 | 0.949646 | ADAMW     | 0.772745    |               | final_validation_auc |
| ADALAYER      | ADAGRAD       | 0.947576 | 0.955074 | ADAGRAD   | 0.250632    |               | final_validation_auc |
| ADALAYER      | NOVOGRAD      | 0.947576 | 0.949761 | NOVOGRAD  | 0.68955     |               | final_validation_auc |
| ADAM_MINI     | SGD           | 0.952349 | 0.952965 | SGD       | 0.679375    |               | final_validation_auc |
| ADAM_MINI     | ADAMW         | 0.952349 | 0.949646 | ADAM_MINI | 0.640066    |               | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.952349 | 0.955074 | ADAGRAD   | 0.472792    |               | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.952349 | 0.949761 | ADAM_MINI | 0.320175    |               | final_validation_auc |
| SGD           | ADAMW         | 0.952965 | 0.949646 | SGD       | 0.576016    |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.952965 | 0.955074 | ADAGRAD   | 0.579356    |               | final_validation_auc |
| SGD           | NOVOGRAD      | 0.952965 | 0.949761 | SGD       | 0.25604     |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.949646 | 0.955074 | ADAGRAD   | 0.414774    |               | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.949646 | 0.949761 | NOVOGRAD  | 0.984307    |               | final_validation_auc |
| ADAGRAD       | NOVOGRAD      | 0.955074 | 0.949761 | ADAGRAD   | 0.234315    |               | final_validation_auc |

