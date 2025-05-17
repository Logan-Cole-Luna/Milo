# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_lr_0.01 | 0.3026 | 0.1211 | 0.0856 | -0.7851 | 1.3903 |
| MILO_lr_0.1 | 0.4349 | 0.0916 | 0.0648 | -0.3885 | 1.2582 |
| MILO_lr_0.001 | 0.2305 | 0.0040 | 0.0028 | 0.1946 | 0.2664 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |   p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:--------------|----------:|:--------------|:----------------------|
| MILO_lr_0.01  | MILO_lr_0.1   | 0.302585 | 0.434851 | MILO_lr_0.01  |  0.350902 |               | final_validation_loss |
| MILO_lr_0.01  | MILO_lr_0.001 | 0.302585 | 0.230522 | MILO_lr_0.001 |  0.554474 |               | final_validation_loss |
| MILO_lr_0.1   | MILO_lr_0.001 | 0.434851 | 0.230522 | MILO_lr_0.001 |  0.194925 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_lr_0.01 | 89.4670 | 4.3346 | 3.0650 | 50.5225 | 128.4115 |
| MILO_lr_0.1 | 84.8920 | 3.2753 | 2.3160 | 55.4644 | 114.3196 |
| MILO_lr_0.001 | 91.9960 | 0.1273 | 0.0900 | 90.8524 | 93.1396 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:--------------|----------:|:--------------|:--------------------------|
| MILO_lr_0.01  | MILO_lr_0.1   |   89.467 |   84.892 | MILO_lr_0.01  |  0.363515 |               | final_validation_accuracy |
| MILO_lr_0.01  | MILO_lr_0.001 |   89.467 |   91.996 | MILO_lr_0.001 |  0.560756 |               | final_validation_accuracy |
| MILO_lr_0.1   | MILO_lr_0.001 |   84.892 |   91.996 | MILO_lr_0.001 |  0.200167 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_lr_0.01 | 0.8945 | 0.0435 | 0.0308 | 0.5037 | 1.2854 |
| MILO_lr_0.1 | 0.8492 | 0.0332 | 0.0235 | 0.5507 | 1.1477 |
| MILO_lr_0.001 | 0.9198 | 0.0013 | 0.0009 | 0.9085 | 0.9311 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:--------------|----------:|:--------------|:--------------------------|
| MILO_lr_0.01  | MILO_lr_0.1   | 0.894532 | 0.849202 | MILO_lr_0.01  |  0.369209 |               | final_validation_f1_score |
| MILO_lr_0.01  | MILO_lr_0.001 | 0.894532 | 0.919817 | MILO_lr_0.001 |  0.561949 |               | final_validation_f1_score |
| MILO_lr_0.1   | MILO_lr_0.001 | 0.849202 | 0.919817 | MILO_lr_0.001 |  0.204031 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_lr_0.01 | 0.9932 | 0.0049 | 0.0035 | 0.9492 | 1.0371 |
| MILO_lr_0.1 | 0.9888 | 0.0060 | 0.0042 | 0.9353 | 1.0424 |
| MILO_lr_0.001 | 0.9959 | 0.0001 | 0.0001 | 0.9949 | 0.9969 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:--------------|----------:|:--------------|:---------------------|
| MILO_lr_0.01  | MILO_lr_0.1   | 0.993174 | 0.98882  | MILO_lr_0.01  |  0.510914 |               | final_validation_auc |
| MILO_lr_0.01  | MILO_lr_0.001 | 0.993174 | 0.995926 | MILO_lr_0.001 |  0.572015 |               | final_validation_auc |
| MILO_lr_0.1   | MILO_lr_0.001 | 0.98882  | 0.995926 | MILO_lr_0.001 |  0.34061  |               | final_validation_auc |

