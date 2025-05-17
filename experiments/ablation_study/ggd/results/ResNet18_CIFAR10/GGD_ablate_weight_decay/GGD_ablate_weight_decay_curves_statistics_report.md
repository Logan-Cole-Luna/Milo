# Statistical Analysis Report

## Methodology

Standard deviations, standard errors, and 95% confidence intervals were computed using the t-distribution.

### Validation_Loss Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_weight_decay_0.0001 | 0.4316 | 0.0029 | 0.0017 | 0.4243 | 0.4388 |
| milo_weight_decay_0.001 | 0.4171 | 0.0068 | 0.0039 | 0.4002 | 0.4339 |
| milo_weight_decay_0.0 | 0.4368 | 0.0180 | 0.0104 | 0.3922 | 0.4815 |

#### Pairwise Significance Tests

| Optimizer A             | Optimizer B            |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric                |
|:------------------------|:-----------------------|---------:|---------:|:------------------------|----------:|:--------------|:----------------------|
| milo_weight_decay_0.0001 | milo_weight_decay_0.001 | 0.431564 | 0.417069 | milo_weight_decay_0.001  | 0.0494508 | *             | final_validation_loss |
| milo_weight_decay_0.0001 | milo_weight_decay_0.0   | 0.431564 | 0.43682  | milo_weight_decay_0.0001 | 0.664588  |               | final_validation_loss |
| milo_weight_decay_0.001  | milo_weight_decay_0.0   | 0.417069 | 0.43682  | milo_weight_decay_0.001  | 0.18867   |               | final_validation_loss |

### Validation_Accuracy Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_weight_decay_0.0001 | 85.4633 | 0.2136 | 0.1233 | 84.9327 | 85.9940 |
| milo_weight_decay_0.001 | 85.9867 | 0.3035 | 0.1752 | 85.2326 | 86.7407 |
| milo_weight_decay_0.0 | 85.4267 | 0.6872 | 0.3968 | 83.7196 | 87.1337 |

#### Pairwise Significance Tests

| Optimizer A             | Optimizer B            |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric                    |
|:------------------------|:-----------------------|---------:|---------:|:------------------------|----------:|:--------------|:--------------------------|
| milo_weight_decay_0.0001 | milo_weight_decay_0.001 |  85.4633 |  85.9867 | milo_weight_decay_0.001  | 0.0782881 |               | final_validation_accuracy |
| milo_weight_decay_0.0001 | milo_weight_decay_0.0   |  85.4633 |  85.4267 | milo_weight_decay_0.0001 | 0.936549  |               | final_validation_accuracy |
| milo_weight_decay_0.001  | milo_weight_decay_0.0   |  85.9867 |  85.4267 | milo_weight_decay_0.001  | 0.29443   |               | final_validation_accuracy |

### Validation_F1_Score Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_weight_decay_0.0001 | 0.8539 | 0.0016 | 0.0009 | 0.8500 | 0.8578 |
| milo_weight_decay_0.001 | 0.8593 | 0.0030 | 0.0018 | 0.8518 | 0.8669 |
| milo_weight_decay_0.0 | 0.8540 | 0.0070 | 0.0040 | 0.8367 | 0.8713 |

#### Pairwise Significance Tests

| Optimizer A             | Optimizer B            |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric                    |
|:------------------------|:-----------------------|---------:|---------:|:-----------------------|----------:|:--------------|:--------------------------|
| milo_weight_decay_0.0001 | milo_weight_decay_0.001 | 0.853853 | 0.859319 | milo_weight_decay_0.001 | 0.0695066 |               | final_validation_f1_score |
| milo_weight_decay_0.0001 | milo_weight_decay_0.0   | 0.853853 | 0.853995 | milo_weight_decay_0.0   | 0.975322  |               | final_validation_f1_score |
| milo_weight_decay_0.001  | milo_weight_decay_0.0   | 0.859319 | 0.853995 | milo_weight_decay_0.001 | 0.319687  |               | final_validation_f1_score |

### Validation_Auc Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_weight_decay_0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_weight_decay_0.001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_weight_decay_0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Pairwise Significance Tests

| Optimizer A             | Optimizer B            |   Mean A |   Mean B | Better                 |   p-value | Significant   | Metric               |
|:------------------------|:-----------------------|---------:|---------:|:-----------------------|----------:|:--------------|:---------------------|
| milo_weight_decay_0.0001 | milo_weight_decay_0.001 |        0 |        0 | milo_weight_decay_0.001 |       nan |               | final_validation_auc |
| milo_weight_decay_0.0001 | milo_weight_decay_0.0   |        0 |        0 | milo_weight_decay_0.0   |       nan |               | final_validation_auc |
| milo_weight_decay_0.001  | milo_weight_decay_0.0   |        0 |        0 | milo_weight_decay_0.0   |       nan |               | final_validation_auc |

