# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_momentum_0.9 | 0.1767 | 0.0185 | 0.0131 | 0.0107 | 0.3427 |
| MILO_momentum_0.0 | 0.1438 | 0.0037 | 0.0027 | 0.1101 | 0.1774 |
| MILO_momentum_0.45 | 0.1244 | 0.0071 | 0.0050 | 0.0607 | 0.1881 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric                |
|:------------------|:-------------------|---------:|---------:|:-------------------|----------:|:--------------|:----------------------|
| MILO_momentum_0.9 | MILO_momentum_0.0  | 0.176711 | 0.143759 | MILO_momentum_0.0  |  0.229274 |               | final_validation_loss |
| MILO_momentum_0.9 | MILO_momentum_0.45 | 0.176711 | 0.124386 | MILO_momentum_0.45 |  0.121386 |               | final_validation_loss |
| MILO_momentum_0.0 | MILO_momentum_0.45 | 0.143759 | 0.124386 | MILO_momentum_0.45 |  0.110202 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_momentum_0.9 | 93.8540 | 0.5572 | 0.3940 | 88.8478 | 98.8602 |
| MILO_momentum_0.0 | 94.9970 | 0.1598 | 0.1130 | 93.5612 | 96.4328 |
| MILO_momentum_0.45 | 95.7150 | 0.1739 | 0.1230 | 94.1521 | 97.2779 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric                    |
|:------------------|:-------------------|---------:|---------:|:-------------------|----------:|:--------------|:--------------------------|
| MILO_momentum_0.9 | MILO_momentum_0.0  |   93.854 |   94.997 | MILO_momentum_0.0  | 0.190044  |               | final_validation_accuracy |
| MILO_momentum_0.9 | MILO_momentum_0.45 |   93.854 |   95.715 | MILO_momentum_0.45 | 0.108318  |               | final_validation_accuracy |
| MILO_momentum_0.0 | MILO_momentum_0.45 |   94.997 |   95.715 | MILO_momentum_0.45 | 0.0507217 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_momentum_0.9 | 0.9385 | 0.0059 | 0.0042 | 0.8858 | 0.9913 |
| MILO_momentum_0.0 | 0.9499 | 0.0015 | 0.0011 | 0.9361 | 0.9638 |
| MILO_momentum_0.45 | 0.9571 | 0.0018 | 0.0012 | 0.9413 | 0.9729 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric                    |
|:------------------|:-------------------|---------:|---------:|:-------------------|----------:|:--------------|:--------------------------|
| MILO_momentum_0.9 | MILO_momentum_0.0  | 0.938537 | 0.949929 | MILO_momentum_0.0  | 0.204509  |               | final_validation_f1_score |
| MILO_momentum_0.9 | MILO_momentum_0.45 | 0.938537 | 0.95712  | MILO_momentum_0.45 | 0.116876  |               | final_validation_f1_score |
| MILO_momentum_0.0 | MILO_momentum_0.45 | 0.949929 | 0.95712  | MILO_momentum_0.45 | 0.0505691 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_momentum_0.9 | 0.9975 | 0.0004 | 0.0003 | 0.9935 | 1.0015 |
| MILO_momentum_0.0 | 0.9983 | 0.0001 | 0.0001 | 0.9975 | 0.9990 |
| MILO_momentum_0.45 | 0.9987 | 0.0001 | 0.0001 | 0.9975 | 0.9999 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric               |
|:------------------|:-------------------|---------:|---------:|:-------------------|----------:|:--------------|:---------------------|
| MILO_momentum_0.9 | MILO_momentum_0.0  | 0.997538 | 0.998269 | MILO_momentum_0.0  | 0.248886  |               | final_validation_auc |
| MILO_momentum_0.9 | MILO_momentum_0.45 | 0.997538 | 0.99867  | MILO_momentum_0.45 | 0.149394  |               | final_validation_auc |
| MILO_momentum_0.0 | MILO_momentum_0.45 | 0.998269 | 0.99867  | MILO_momentum_0.45 | 0.0859353 |               | final_validation_auc |

