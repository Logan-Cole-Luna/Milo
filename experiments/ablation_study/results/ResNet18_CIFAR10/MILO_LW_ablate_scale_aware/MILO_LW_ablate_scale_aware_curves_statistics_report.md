# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_aware_True | 3.6711 | 1.8643 | 1.3182 | -13.0788 | 20.4210 |
| MILO_LW_scale_aware_False | 1.1395 | 0.3640 | 0.2574 | -2.1306 | 4.4096 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric                |
|:-------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:----------------------|
| MILO_LW_scale_aware_True | MILO_LW_scale_aware_False |  3.67111 |  1.13946 | MILO_LW_scale_aware_False |  0.296773 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_aware_True | 95.7010 | 0.0523 | 0.0370 | 95.2309 | 96.1711 |
| MILO_LW_scale_aware_False | 96.0930 | 0.1598 | 0.1130 | 94.6572 | 97.5288 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric                    |
|:-------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:--------------------------|
| MILO_LW_scale_aware_True | MILO_LW_scale_aware_False |   95.701 |   96.093 | MILO_LW_scale_aware_False |  0.151577 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_aware_True | 0.9570 | 0.0004 | 0.0003 | 0.9530 | 0.9610 |
| MILO_LW_scale_aware_False | 0.9609 | 0.0016 | 0.0011 | 0.9465 | 0.9754 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric                    |
|:-------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:--------------------------|
| MILO_LW_scale_aware_True | MILO_LW_scale_aware_False | 0.957033 | 0.960937 | MILO_LW_scale_aware_False |  0.159494 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_scale_aware_True | 0.9918 | 0.0037 | 0.0026 | 0.9587 | 1.0249 |
| MILO_LW_scale_aware_False | 0.9974 | 0.0007 | 0.0005 | 0.9907 | 1.0041 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric               |
|:-------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:---------------------|
| MILO_LW_scale_aware_True | MILO_LW_scale_aware_False | 0.991824 | 0.997396 | MILO_LW_scale_aware_False |  0.268492 |               | final_validation_auc |

