# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_momentum_0.9 | 0.1064 | 0.0083 | 0.0059 | 0.0319 | 0.1809 |
| MILO_LW_momentum_0.0 | 0.1519 | 0.0195 | 0.0138 | -0.0231 | 0.3268 |
| MILO_LW_momentum_0.45 | 0.1885 | 0.0048 | 0.0034 | 0.1456 | 0.2314 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric                |
|:---------------------|:----------------------|---------:|---------:|:---------------------|----------:|:--------------|:----------------------|
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.0  | 0.10637  | 0.151873 | MILO_LW_momentum_0.9 | 0.147493  |               | final_validation_loss |
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.45 | 0.10637  | 0.188516 | MILO_LW_momentum_0.9 | 0.0145723 | *             | final_validation_loss |
| MILO_LW_momentum_0.0 | MILO_LW_momentum_0.45 | 0.151873 | 0.188516 | MILO_LW_momentum_0.0 | 0.21296   |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_momentum_0.9 | 96.7600 | 0.2121 | 0.1500 | 94.8541 | 98.6659 |
| MILO_LW_momentum_0.0 | 95.7475 | 0.3241 | 0.2292 | 92.8357 | 98.6593 |
| MILO_LW_momentum_0.45 | 94.9267 | 0.2333 | 0.1650 | 92.8301 | 97.0232 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric                    |
|:---------------------|:----------------------|---------:|---------:|:---------------------|----------:|:--------------|:--------------------------|
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.0  |  96.76   |  95.7475 | MILO_LW_momentum_0.9 | 0.0823542 |               | final_validation_accuracy |
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.45 |  96.76   |  94.9267 | MILO_LW_momentum_0.9 | 0.0148622 | *             | final_validation_accuracy |
| MILO_LW_momentum_0.0 | MILO_LW_momentum_0.45 |  95.7475 |  94.9267 | MILO_LW_momentum_0.0 | 0.112515  |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_momentum_0.9 | 0.9689 | 0.0019 | 0.0014 | 0.9515 | 0.9864 |
| MILO_LW_momentum_0.0 | 0.9598 | 0.0029 | 0.0021 | 0.9337 | 0.9859 |
| MILO_LW_momentum_0.45 | 0.9525 | 0.0020 | 0.0014 | 0.9346 | 0.9704 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric                    |
|:---------------------|:----------------------|---------:|---------:|:---------------------|----------:|:--------------|:--------------------------|
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.0  | 0.968933 | 0.959776 | MILO_LW_momentum_0.9 | 0.0804937 |               | final_validation_f1_score |
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.45 | 0.968933 | 0.952525 | MILO_LW_momentum_0.9 | 0.0141197 | *             | final_validation_f1_score |
| MILO_LW_momentum_0.0 | MILO_LW_momentum_0.45 | 0.959776 | 0.952525 | MILO_LW_momentum_0.0 | 0.115485  |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_momentum_0.9 | 0.9991 | 0.0001 | 0.0001 | 0.9980 | 1.0001 |
| MILO_LW_momentum_0.0 | 0.9984 | 0.0002 | 0.0002 | 0.9963 | 1.0005 |
| MILO_LW_momentum_0.45 | 0.9978 | 0.0002 | 0.0001 | 0.9962 | 0.9993 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric               |
|:---------------------|:----------------------|---------:|---------:|:---------------------|----------:|:--------------|:---------------------|
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.0  | 0.999062 | 0.998409 | MILO_LW_momentum_0.9 | 0.111475  |               | final_validation_auc |
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.45 | 0.999062 | 0.997766 | MILO_LW_momentum_0.9 | 0.020103  | *             | final_validation_auc |
| MILO_LW_momentum_0.0 | MILO_LW_momentum_0.45 | 0.998409 | 0.997766 | MILO_LW_momentum_0.0 | 0.0989981 |               | final_validation_auc |

