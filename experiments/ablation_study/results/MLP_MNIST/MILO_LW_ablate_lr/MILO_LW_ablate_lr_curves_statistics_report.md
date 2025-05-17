# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_lr_0.01 | 0.1174 | 0.0047 | 0.0033 | 0.0752 | 0.1597 |
| MILO_LW_lr_0.1 | 0.0854 | 0.0037 | 0.0026 | 0.0519 | 0.1188 |
| MILO_LW_lr_0.001 | 0.0848 | 0.0002 | 0.0002 | 0.0826 | 0.0870 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A     | Optimizer B      |   Mean A |    Mean B | Better           |   p-value | Significant   | Metric                |
|:----------------|:-----------------|---------:|----------:|:-----------------|----------:|:--------------|:----------------------|
| MILO_LW_lr_0.01 | MILO_LW_lr_0.1   | 0.117433 | 0.085355  | MILO_LW_lr_0.1   | 0.0196194 | *             | final_validation_loss |
| MILO_LW_lr_0.01 | MILO_LW_lr_0.001 | 0.117433 | 0.0847776 | MILO_LW_lr_0.001 | 0.0639464 |               | final_validation_loss |
| MILO_LW_lr_0.1  | MILO_LW_lr_0.001 | 0.085355 | 0.0847776 | MILO_LW_lr_0.001 | 0.862425  |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_lr_0.01 | 96.8267 | 0.2428 | 0.1717 | 94.6454 | 99.0079 |
| MILO_LW_lr_0.1 | 97.3958 | 0.2722 | 0.1925 | 94.9499 | 99.8418 |
| MILO_LW_lr_0.001 | 97.2000 | 0.0047 | 0.0033 | 97.1576 | 97.2424 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A     | Optimizer B      |   Mean A |   Mean B | Better           |   p-value | Significant   | Metric                    |
|:----------------|:-----------------|---------:|---------:|:-----------------|----------:|:--------------|:--------------------------|
| MILO_LW_lr_0.01 | MILO_LW_lr_0.1   |  96.8267 |  97.3958 | MILO_LW_lr_0.1   |  0.15971  |               | final_validation_accuracy |
| MILO_LW_lr_0.01 | MILO_LW_lr_0.001 |  96.8267 |  97.2    | MILO_LW_lr_0.001 |  0.274275 |               | final_validation_accuracy |
| MILO_LW_lr_0.1  | MILO_LW_lr_0.001 |  97.3958 |  97.2    | MILO_LW_lr_0.1   |  0.494503 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_lr_0.01 | 0.9695 | 0.0021 | 0.0015 | 0.9502 | 0.9887 |
| MILO_LW_lr_0.1 | 0.9748 | 0.0025 | 0.0018 | 0.9522 | 0.9973 |
| MILO_LW_lr_0.001 | 0.9730 | 0.0000 | 0.0000 | 0.9727 | 0.9733 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A     | Optimizer B      |   Mean A |   Mean B | Better           |   p-value | Significant   | Metric                    |
|:----------------|:-----------------|---------:|---------:|:-----------------|----------:|:--------------|:--------------------------|
| MILO_LW_lr_0.01 | MILO_LW_lr_0.1   | 0.969452 | 0.974773 | MILO_LW_lr_0.1   |  0.153289 |               | final_validation_f1_score |
| MILO_LW_lr_0.01 | MILO_LW_lr_0.001 | 0.969452 | 0.97299  | MILO_LW_lr_0.001 |  0.257574 |               | final_validation_f1_score |
| MILO_LW_lr_0.1  | MILO_LW_lr_0.001 | 0.974773 | 0.97299  | MILO_LW_lr_0.1   |  0.498422 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_lr_0.01 | 0.9990 | 0.0001 | 0.0001 | 0.9980 | 1.0000 |
| MILO_LW_lr_0.1 | 0.9994 | 0.0001 | 0.0001 | 0.9984 | 1.0003 |
| MILO_LW_lr_0.001 | 0.9993 | 0.0000 | 0.0000 | 0.9993 | 0.9993 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A     | Optimizer B      |   Mean A |   Mean B | Better           |   p-value | Significant   | Metric               |
|:----------------|:-----------------|---------:|---------:|:-----------------|----------:|:--------------|:---------------------|
| MILO_LW_lr_0.01 | MILO_LW_lr_0.1   | 0.998985 | 0.999362 | MILO_LW_lr_0.1   | 0.0718987 |               | final_validation_auc |
| MILO_LW_lr_0.01 | MILO_LW_lr_0.001 | 0.998985 | 0.999296 | MILO_LW_lr_0.001 | 0.156338  |               | final_validation_auc |
| MILO_LW_lr_0.1  | MILO_LW_lr_0.001 | 0.999362 | 0.999296 | MILO_LW_lr_0.1   | 0.530185  |               | final_validation_auc |

