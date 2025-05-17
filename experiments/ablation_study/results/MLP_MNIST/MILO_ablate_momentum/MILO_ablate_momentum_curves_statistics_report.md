# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_momentum_0.9 | 0.0005 | 0.0002 | 0.0002 | -0.0015 | 0.0024 |
| MILO_momentum_0.0 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0004 |
| MILO_momentum_0.45 | 0.0002 | 0.0000 | 0.0000 | 0.0002 | 0.0002 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B        |      Mean A |      Mean B | Better             |   p-value | Significant   | Metric                |
|:------------------|:-------------------|------------:|------------:|:-------------------|----------:|:--------------|:----------------------|
| MILO_momentum_0.9 | MILO_momentum_0.0  | 0.000467913 | 0.000230878 | MILO_momentum_0.0  |  0.36645  |               | final_validation_loss |
| MILO_momentum_0.9 | MILO_momentum_0.45 | 0.000467913 | 0.000210112 | MILO_momentum_0.45 |  0.344237 |               | final_validation_loss |
| MILO_momentum_0.0 | MILO_momentum_0.45 | 0.000230878 | 0.000210112 | MILO_momentum_0.45 |  0.393586 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_momentum_0.9 | 99.9833 | 0.0047 | 0.0033 | 99.9410 | 100.0257 |
| MILO_momentum_0.0 | 99.9908 | 0.0012 | 0.0008 | 99.9802 | 100.0014 |
| MILO_momentum_0.45 | 99.9900 | 0.0000 | 0.0000 | 99.9900 | 99.9900 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric                    |
|:------------------|:-------------------|---------:|---------:|:-------------------|----------:|:--------------|:--------------------------|
| MILO_momentum_0.9 | MILO_momentum_0.0  |  99.9833 |  99.9908 | MILO_momentum_0.0  |  0.25098  |               | final_validation_accuracy |
| MILO_momentum_0.9 | MILO_momentum_0.45 |  99.9833 |  99.99   | MILO_momentum_0.45 |  0.295167 |               | final_validation_accuracy |
| MILO_momentum_0.0 | MILO_momentum_0.45 |  99.9908 |  99.99   | MILO_momentum_0.0  |  0.5      |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_momentum_0.9 | 0.9998 | 0.0000 | 0.0000 | 0.9994 | 1.0003 |
| MILO_momentum_0.0 | 0.9999 | 0.0000 | 0.0000 | 0.9998 | 1.0000 |
| MILO_momentum_0.45 | 0.9999 | 0.0000 | 0.0000 | 0.9999 | 0.9999 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric                    |
|:------------------|:-------------------|---------:|---------:|:-------------------|----------:|:--------------|:--------------------------|
| MILO_momentum_0.9 | MILO_momentum_0.0  | 0.999832 | 0.999908 | MILO_momentum_0.0  |  0.257979 |               | final_validation_f1_score |
| MILO_momentum_0.9 | MILO_momentum_0.45 | 0.999832 | 0.9999   | MILO_momentum_0.45 |  0.300466 |               | final_validation_f1_score |
| MILO_momentum_0.0 | MILO_momentum_0.45 | 0.999908 | 0.9999   | MILO_momentum_0.0  |  0.51241  |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_momentum_0.9 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| MILO_momentum_0.0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| MILO_momentum_0.45 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B        |   Mean A |   Mean B | Better             |   p-value | Significant   | Metric               |
|:------------------|:-------------------|---------:|---------:|:-------------------|----------:|:--------------|:---------------------|
| MILO_momentum_0.9 | MILO_momentum_0.0  |        1 |        1 | MILO_momentum_0.0  |  0.399452 |               | final_validation_auc |
| MILO_momentum_0.9 | MILO_momentum_0.45 |        1 |        1 | MILO_momentum_0.45 |  0.394915 |               | final_validation_auc |
| MILO_momentum_0.0 | MILO_momentum_0.45 |        1 |        1 | MILO_momentum_0.45 |  0.80604  |               | final_validation_auc |

