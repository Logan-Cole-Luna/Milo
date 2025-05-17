# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_weight_decay_0.0001 | 0.0002 | 0.0000 | 0.0000 | -0.0001 | 0.0005 |
| MILO_weight_decay_0.001 | 0.0002 | 0.0001 | 0.0000 | -0.0003 | 0.0006 |
| MILO_weight_decay_0.0 | 0.0001 | 0.0000 | 0.0000 | -0.0002 | 0.0004 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B             |      Mean A |      Mean B | Better                  |   p-value | Significant   | Metric                |
|:-------------------------|:------------------------|------------:|------------:|:------------------------|----------:|:--------------|:----------------------|
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.001 | 0.000196002 | 0.000158743 | MILO_weight_decay_0.001 | 0.501798  |               | final_validation_loss |
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.0   | 0.000196002 | 8.86774e-05 | MILO_weight_decay_0.0   | 0.0912608 |               | final_validation_loss |
| MILO_weight_decay_0.001  | MILO_weight_decay_0.0   | 0.000158743 | 8.86774e-05 | MILO_weight_decay_0.0   | 0.268635  |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_weight_decay_0.0001 | 99.9908 | 0.0035 | 0.0025 | 99.9591 | 100.0226 |
| MILO_weight_decay_0.001 | 99.9933 | 0.0024 | 0.0017 | 99.9722 | 100.0145 |
| MILO_weight_decay_0.0 | 99.9958 | 0.0035 | 0.0025 | 99.9641 | 100.0276 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B             |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric                    |
|:-------------------------|:------------------------|---------:|---------:|:------------------------|----------:|:--------------|:--------------------------|
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.001 |  99.9908 |  99.9933 | MILO_weight_decay_0.001 |  0.503838 |               | final_validation_accuracy |
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.0   |  99.9908 |  99.9958 | MILO_weight_decay_0.0   |  0.292893 |               | final_validation_accuracy |
| MILO_weight_decay_0.001  | MILO_weight_decay_0.0   |  99.9933 |  99.9958 | MILO_weight_decay_0.0   |  0.503838 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_weight_decay_0.0001 | 0.9999 | 0.0000 | 0.0000 | 0.9996 | 1.0002 |
| MILO_weight_decay_0.001 | 0.9999 | 0.0000 | 0.0000 | 0.9997 | 1.0001 |
| MILO_weight_decay_0.0 | 1.0000 | 0.0000 | 0.0000 | 0.9996 | 1.0003 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B             |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric                    |
|:-------------------------|:------------------------|---------:|---------:|:------------------------|----------:|:--------------|:--------------------------|
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.001 | 0.999908 | 0.999934 | MILO_weight_decay_0.001 |  0.481041 |               | final_validation_f1_score |
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.0   | 0.999908 | 0.999959 | MILO_weight_decay_0.0   |  0.28085  |               | final_validation_f1_score |
| MILO_weight_decay_0.001  | MILO_weight_decay_0.0   | 0.999934 | 0.999959 | MILO_weight_decay_0.0   |  0.500342 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_weight_decay_0.0001 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| MILO_weight_decay_0.001 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| MILO_weight_decay_0.0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B             |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric               |
|:-------------------------|:------------------------|---------:|---------:|:------------------------|----------:|:--------------|:---------------------|
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.001 |        1 |        1 | MILO_weight_decay_0.001 |  0.435207 |               | final_validation_auc |
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.0   |        1 |        1 | MILO_weight_decay_0.0   |  0.156618 |               | final_validation_auc |
| MILO_weight_decay_0.001  | MILO_weight_decay_0.0   |        1 |        1 | MILO_weight_decay_0.0   |  0.267882 |               | final_validation_auc |

