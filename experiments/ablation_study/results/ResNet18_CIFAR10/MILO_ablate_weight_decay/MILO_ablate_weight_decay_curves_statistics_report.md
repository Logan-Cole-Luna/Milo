# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_weight_decay_0.0001 | 0.0955 | 0.0124 | 0.0088 | -0.0164 | 0.2073 |
| MILO_weight_decay_0.001 | 0.0658 | 0.0085 | 0.0060 | -0.0110 | 0.1426 |
| MILO_weight_decay_0.0 | 0.0448 | 0.0073 | 0.0052 | -0.0210 | 0.1107 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B             |    Mean A |    Mean B | Better                  |   p-value | Significant   | Metric                |
|:-------------------------|:------------------------|----------:|----------:|:------------------------|----------:|:--------------|:----------------------|
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.001 | 0.0954654 | 0.0658204 | MILO_weight_decay_0.001 | 0.12425   |               | final_validation_loss |
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.0   | 0.0954654 | 0.0448443 | MILO_weight_decay_0.0   | 0.0576936 |               | final_validation_loss |
| MILO_weight_decay_0.001  | MILO_weight_decay_0.0   | 0.0658204 | 0.0448443 | MILO_weight_decay_0.0   | 0.121747  |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_weight_decay_0.0001 | 96.6260 | 0.3564 | 0.2520 | 93.4240 | 99.8280 |
| MILO_weight_decay_0.001 | 97.6530 | 0.1881 | 0.1330 | 95.9631 | 99.3429 |
| MILO_weight_decay_0.0 | 98.4010 | 0.2079 | 0.1470 | 96.5332 | 100.2688 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B             |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric                    |
|:-------------------------|:------------------------|---------:|---------:|:------------------------|----------:|:--------------|:--------------------------|
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.001 |   96.626 |   97.653 | MILO_weight_decay_0.001 | 0.102346  |               | final_validation_accuracy |
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.0   |   96.626 |   98.401 | MILO_weight_decay_0.0   | 0.0424151 | *             | final_validation_accuracy |
| MILO_weight_decay_0.001  | MILO_weight_decay_0.0   |   97.653 |   98.401 | MILO_weight_decay_0.0   | 0.064596  |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_weight_decay_0.0001 | 0.9663 | 0.0036 | 0.0025 | 0.9343 | 0.9983 |
| MILO_weight_decay_0.001 | 0.9766 | 0.0019 | 0.0014 | 0.9593 | 0.9938 |
| MILO_weight_decay_0.0 | 0.9840 | 0.0020 | 0.0014 | 0.9657 | 1.0023 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B             |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric                    |
|:-------------------------|:------------------------|---------:|---------:|:------------------------|----------:|:--------------|:--------------------------|
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.001 | 0.966276 | 0.976569 | MILO_weight_decay_0.001 | 0.101004  |               | final_validation_f1_score |
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.0   | 0.966276 | 0.984027 | MILO_weight_decay_0.0   | 0.0431756 | *             | final_validation_f1_score |
| MILO_weight_decay_0.001  | MILO_weight_decay_0.0   | 0.976569 | 0.984027 | MILO_weight_decay_0.0   | 0.0640281 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_weight_decay_0.0001 | 0.9992 | 0.0002 | 0.0001 | 0.9974 | 1.0011 |
| MILO_weight_decay_0.001 | 0.9996 | 0.0001 | 0.0001 | 0.9988 | 1.0004 |
| MILO_weight_decay_0.0 | 0.9998 | 0.0001 | 0.0000 | 0.9994 | 1.0003 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A              | Optimizer B             |   Mean A |   Mean B | Better                  |   p-value | Significant   | Metric               |
|:-------------------------|:------------------------|---------:|---------:|:------------------------|----------:|:--------------|:---------------------|
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.001 | 0.999206 | 0.999605 | MILO_weight_decay_0.001 |  0.182589 |               | final_validation_auc |
| MILO_weight_decay_0.0001 | MILO_weight_decay_0.0   | 0.999206 | 0.999816 | MILO_weight_decay_0.0   |  0.131887 |               | final_validation_auc |
| MILO_weight_decay_0.001  | MILO_weight_decay_0.0   | 0.999605 | 0.999816 | MILO_weight_decay_0.0   |  0.135577 |               | final_validation_auc |

