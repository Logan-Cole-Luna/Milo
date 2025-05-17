# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_lr_0.01 | 6.5581 | 3.7581 | 2.6574 | -27.2069 | 40.3232 |
| MILO_LW_lr_0.1 | 4.1373 | 0.2016 | 0.1426 | 2.3260 | 5.9485 |
| MILO_LW_lr_0.001 | 1.3520 | 0.1238 | 0.0875 | 0.2398 | 2.4642 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A     | Optimizer B      |   Mean A |   Mean B | Better           |    p-value | Significant   | Metric                |
|:----------------|:-----------------|---------:|---------:|:-----------------|-----------:|:--------------|:----------------------|
| MILO_LW_lr_0.01 | MILO_LW_lr_0.1   |  6.55814 |  4.13725 | MILO_LW_lr_0.1   | 0.529381   |               | final_validation_loss |
| MILO_LW_lr_0.01 | MILO_LW_lr_0.001 |  6.55814 |  1.352   | MILO_LW_lr_0.001 | 0.300179   |               | final_validation_loss |
| MILO_LW_lr_0.1  | MILO_LW_lr_0.001 |  4.13725 |  1.352   | MILO_LW_lr_0.001 | 0.00763509 | **            | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_lr_0.01 | 90.1880 | 0.4016 | 0.2840 | 86.5794 | 93.7966 |
| MILO_LW_lr_0.1 | 90.0130 | 1.2459 | 0.8810 | 78.8188 | 101.2072 |
| MILO_LW_lr_0.001 | 94.7850 | 0.0071 | 0.0050 | 94.7215 | 94.8485 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A     | Optimizer B      |   Mean A |   Mean B | Better           |   p-value | Significant   | Metric                    |
|:----------------|:-----------------|---------:|---------:|:-----------------|----------:|:--------------|:--------------------------|
| MILO_LW_lr_0.01 | MILO_LW_lr_0.1   |   90.188 |   90.013 | MILO_LW_lr_0.01  |  0.876825 |               | final_validation_accuracy |
| MILO_LW_lr_0.01 | MILO_LW_lr_0.001 |   90.188 |   94.785 | MILO_LW_lr_0.001 |  0.039223 | *             | final_validation_accuracy |
| MILO_LW_lr_0.1  | MILO_LW_lr_0.001 |   90.013 |   94.785 | MILO_LW_lr_0.001 |  0.116214 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_lr_0.01 | 0.9021 | 0.0042 | 0.0029 | 0.8648 | 0.9395 |
| MILO_LW_lr_0.1 | 0.8973 | 0.0173 | 0.0122 | 0.7421 | 1.0526 |
| MILO_LW_lr_0.001 | 0.9479 | 0.0001 | 0.0000 | 0.9474 | 0.9484 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A     | Optimizer B      |   Mean A |   Mean B | Better           |   p-value | Significant   | Metric                    |
|:----------------|:-----------------|---------:|---------:|:-----------------|----------:|:--------------|:--------------------------|
| MILO_LW_lr_0.01 | MILO_LW_lr_0.1   | 0.902135 | 0.897336 | MILO_LW_lr_0.01  | 0.762502  |               | final_validation_f1_score |
| MILO_LW_lr_0.01 | MILO_LW_lr_0.001 | 0.902135 | 0.947931 | MILO_LW_lr_0.001 | 0.0407995 | *             | final_validation_f1_score |
| MILO_LW_lr_0.1  | MILO_LW_lr_0.001 | 0.897336 | 0.947931 | MILO_LW_lr_0.001 | 0.150845  |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_lr_0.01 | 0.9833 | 0.0073 | 0.0052 | 0.9173 | 1.0493 |
| MILO_LW_lr_0.1 | 0.9911 | 0.0014 | 0.0010 | 0.9788 | 1.0034 |
| MILO_LW_lr_0.001 | 0.9965 | 0.0002 | 0.0001 | 0.9947 | 0.9984 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A     | Optimizer B      |   Mean A |   Mean B | Better           |   p-value | Significant   | Metric               |
|:----------------|:-----------------|---------:|---------:|:-----------------|----------:|:--------------|:---------------------|
| MILO_LW_lr_0.01 | MILO_LW_lr_0.1   | 0.983296 | 0.991087 | MILO_LW_lr_0.1   |  0.368053 |               | final_validation_auc |
| MILO_LW_lr_0.01 | MILO_LW_lr_0.001 | 0.983296 | 0.996515 | MILO_LW_lr_0.001 |  0.238194 |               | final_validation_auc |
| MILO_LW_lr_0.1  | MILO_LW_lr_0.001 | 0.991087 | 0.996515 | MILO_LW_lr_0.001 |  0.106242 |               | final_validation_auc |

