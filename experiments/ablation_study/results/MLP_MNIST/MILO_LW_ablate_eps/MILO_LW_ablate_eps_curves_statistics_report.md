# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_eps_1e-05 | 0.1430 | 0.0032 | 0.0022 | 0.1146 | 0.1715 |
| MILO_LW_eps_0.005 | 0.1259 | 0.0034 | 0.0024 | 0.0953 | 0.1566 |
| MILO_LW_eps_0.05 | 0.1112 | 0.0061 | 0.0043 | 0.0562 | 0.1661 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric                |
|:------------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:----------------------|
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.005 | 0.143018 | 0.125914 | MILO_LW_eps_0.005 | 0.0355038 | *             | final_validation_loss |
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.05  | 0.143018 | 0.111162 | MILO_LW_eps_0.05  | 0.0442385 | *             | final_validation_loss |
| MILO_LW_eps_0.005 | MILO_LW_eps_0.05  | 0.125914 | 0.111162 | MILO_LW_eps_0.05  | 0.128354  |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_eps_1e-05 | 95.2717 | 0.1226 | 0.0867 | 94.1705 | 96.3729 |
| MILO_LW_eps_0.005 | 95.6450 | 0.1815 | 0.1283 | 94.0144 | 97.2756 |
| MILO_LW_eps_0.05 | 96.1383 | 0.2357 | 0.1667 | 94.0206 | 98.2560 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric                    |
|:------------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:--------------------------|
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.005 |  95.2717 |  95.645  | MILO_LW_eps_0.005 | 0.154697  |               | final_validation_accuracy |
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.05  |  95.2717 |  96.1383 | MILO_LW_eps_0.05  | 0.0730475 |               | final_validation_accuracy |
| MILO_LW_eps_0.005 | MILO_LW_eps_0.05  |  95.645  |  96.1383 | MILO_LW_eps_0.05  | 0.15181   |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_eps_1e-05 | 0.9556 | 0.0011 | 0.0008 | 0.9459 | 0.9653 |
| MILO_LW_eps_0.005 | 0.9590 | 0.0016 | 0.0011 | 0.9448 | 0.9732 |
| MILO_LW_eps_0.05 | 0.9632 | 0.0021 | 0.0015 | 0.9446 | 0.9818 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric                    |
|:------------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:--------------------------|
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.005 | 0.955606 | 0.95898  | MILO_LW_eps_0.005 | 0.146668  |               | final_validation_f1_score |
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.05  | 0.955606 | 0.963231 | MILO_LW_eps_0.05  | 0.0727066 |               | final_validation_f1_score |
| MILO_LW_eps_0.005 | MILO_LW_eps_0.05  | 0.95898  | 0.963231 | MILO_LW_eps_0.05  | 0.155938  |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_eps_1e-05 | 0.9982 | 0.0001 | 0.0001 | 0.9974 | 0.9991 |
| MILO_LW_eps_0.005 | 0.9985 | 0.0001 | 0.0001 | 0.9975 | 0.9995 |
| MILO_LW_eps_0.05 | 0.9988 | 0.0001 | 0.0001 | 0.9976 | 1.0000 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric               |
|:------------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:---------------------|
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.005 | 0.998237 | 0.998495 | MILO_LW_eps_0.005 | 0.136915  |               | final_validation_auc |
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.05  | 0.998237 | 0.998793 | MILO_LW_eps_0.05  | 0.0469714 | *             | final_validation_auc |
| MILO_LW_eps_0.005 | MILO_LW_eps_0.05  | 0.998495 | 0.998793 | MILO_LW_eps_0.05  | 0.136476  |               | final_validation_auc |

