# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_momentum_0.9 | 0.8884 | 0.2163 | 0.1529 | -1.0549 | 2.8316 |
| MILO_LW_momentum_0.0 | 0.5875 | 0.0503 | 0.0355 | 0.1359 | 1.0390 |
| MILO_LW_momentum_0.45 | 0.5709 | 0.0835 | 0.0590 | -0.1793 | 1.3211 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric                |
|:---------------------|:----------------------|---------:|---------:|:----------------------|----------:|:--------------|:----------------------|
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.0  | 0.888378 | 0.587474 | MILO_LW_momentum_0.0  |  0.287034 |               | final_validation_loss |
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.45 | 0.888378 | 0.570894 | MILO_LW_momentum_0.45 |  0.257183 |               | final_validation_loss |
| MILO_LW_momentum_0.0 | MILO_LW_momentum_0.45 | 0.587474 | 0.570894 | MILO_LW_momentum_0.45 |  0.836509 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_momentum_0.9 | 93.9720 | 0.3281 | 0.2320 | 91.0242 | 96.9198 |
| MILO_LW_momentum_0.0 | 94.4290 | 0.0778 | 0.0550 | 93.7302 | 95.1278 |
| MILO_LW_momentum_0.45 | 93.9340 | 1.0663 | 0.7540 | 84.3535 | 103.5145 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric                    |
|:---------------------|:----------------------|---------:|---------:|:---------------------|----------:|:--------------|:--------------------------|
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.0  |   93.972 |   94.429 | MILO_LW_momentum_0.0 |  0.286279 |               | final_validation_accuracy |
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.45 |   93.972 |   93.934 | MILO_LW_momentum_0.9 |  0.968381 |               | final_validation_accuracy |
| MILO_LW_momentum_0.0 | MILO_LW_momentum_0.45 |   94.429 |   93.934 | MILO_LW_momentum_0.0 |  0.630008 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_momentum_0.9 | 0.9394 | 0.0042 | 0.0030 | 0.9015 | 0.9774 |
| MILO_LW_momentum_0.0 | 0.9443 | 0.0009 | 0.0006 | 0.9363 | 0.9523 |
| MILO_LW_momentum_0.45 | 0.9391 | 0.0112 | 0.0079 | 0.8385 | 1.0397 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric                    |
|:---------------------|:----------------------|---------:|---------:|:---------------------|----------:|:--------------|:--------------------------|
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.0  | 0.939415 | 0.944279 | MILO_LW_momentum_0.0 |  0.341751 |               | final_validation_f1_score |
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.45 | 0.939415 | 0.939079 | MILO_LW_momentum_0.9 |  0.973554 |               | final_validation_f1_score |
| MILO_LW_momentum_0.0 | MILO_LW_momentum_0.45 | 0.944279 | 0.939079 | MILO_LW_momentum_0.0 |  0.62988  |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_momentum_0.9 | 0.9970 | 0.0006 | 0.0004 | 0.9914 | 1.0025 |
| MILO_LW_momentum_0.0 | 0.9975 | 0.0001 | 0.0001 | 0.9966 | 0.9983 |
| MILO_LW_momentum_0.45 | 0.9976 | 0.0001 | 0.0001 | 0.9962 | 0.9989 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A          | Optimizer B           |   Mean A |   Mean B | Better                |   p-value | Significant   | Metric               |
|:---------------------|:----------------------|---------:|---------:|:----------------------|----------:|:--------------|:---------------------|
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.0  | 0.996952 | 0.997485 | MILO_LW_momentum_0.0  |  0.432419 |               | final_validation_auc |
| MILO_LW_momentum_0.9 | MILO_LW_momentum_0.45 | 0.996952 | 0.997591 | MILO_LW_momentum_0.45 |  0.370922 |               | final_validation_auc |
| MILO_LW_momentum_0.0 | MILO_LW_momentum_0.45 | 0.997485 | 0.997591 | MILO_LW_momentum_0.45 |  0.501531 |               | final_validation_auc |

