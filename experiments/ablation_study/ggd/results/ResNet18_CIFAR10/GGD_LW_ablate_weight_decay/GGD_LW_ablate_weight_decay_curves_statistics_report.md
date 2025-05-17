# Statistical Analysis Report

## Methodology

Standard deviations, standard errors, and 95% confidence intervals were computed using the t-distribution.

### Validation_Loss Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_weight_decay_0.0001 | 0.2308 | 0.1383 | 0.0798 | -0.1126 | 0.5743 |
| milo_LW_weight_decay_0.001 | 0.0366 | 0.0160 | 0.0093 | -0.0032 | 0.0765 |
| milo_LW_weight_decay_0.0 | 0.0105 | 0.0027 | 0.0016 | 0.0037 | 0.0172 |

#### Pairwise Significance Tests

| Optimizer A                | Optimizer B               |   Mean A |    Mean B | Better                    |   p-value | Significant   | Metric                |
|:---------------------------|:--------------------------|---------:|----------:|:--------------------------|----------:|:--------------|:----------------------|
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.001 | 0.230849 | 0.036608  | milo_LW_weight_decay_0.001 |  0.133612 |               | final_validation_loss |
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.0   | 0.230849 | 0.0104633 | milo_LW_weight_decay_0.0   |  0.109928 |               | final_validation_loss |
| milo_LW_weight_decay_0.001  | milo_LW_weight_decay_0.0   | 0.036608 | 0.0104633 | milo_LW_weight_decay_0.0   |  0.10202  |               | final_validation_loss |

### Validation_Accuracy Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_weight_decay_0.0001 | 91.9453 | 4.8166 | 2.7809 | 79.9803 | 103.9104 |
| milo_LW_weight_decay_0.001 | 98.7033 | 0.5708 | 0.3296 | 97.2854 | 100.1213 |
| milo_LW_weight_decay_0.0 | 99.6227 | 0.0960 | 0.0554 | 99.3842 | 99.8612 |

#### Pairwise Significance Tests

| Optimizer A                | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric                    |
|:---------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:--------------------------|
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.001 |  91.9453 |  98.7033 | milo_LW_weight_decay_0.001 |  0.133806 |               | final_validation_accuracy |
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.0   |  91.9453 |  99.6227 | milo_LW_weight_decay_0.0   |  0.109919 |               | final_validation_accuracy |
| milo_LW_weight_decay_0.001  | milo_LW_weight_decay_0.0   |  98.7033 |  99.6227 | milo_LW_weight_decay_0.0   |  0.104217 |               | final_validation_accuracy |

### Validation_F1_Score Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_weight_decay_0.0001 | 0.9193 | 0.0480 | 0.0277 | 0.8000 | 1.0386 |
| milo_LW_weight_decay_0.001 | 0.9870 | 0.0057 | 0.0033 | 0.9728 | 1.0013 |
| milo_LW_weight_decay_0.0 | 0.9962 | 0.0010 | 0.0006 | 0.9938 | 0.9986 |

#### Pairwise Significance Tests

| Optimizer A                | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric                    |
|:---------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:--------------------------|
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.001 | 0.919326 | 0.987027 | milo_LW_weight_decay_0.001 |  0.132747 |               | final_validation_f1_score |
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.0   | 0.919326 | 0.996227 | milo_LW_weight_decay_0.0   |  0.109073 |               | final_validation_f1_score |
| milo_LW_weight_decay_0.001  | milo_LW_weight_decay_0.0   | 0.987027 | 0.996227 | milo_LW_weight_decay_0.0   |  0.104968 |               | final_validation_f1_score |

### Validation_Auc Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_weight_decay_0.0001 | 0.9955 | 0.0048 | 0.0028 | 0.9835 | 1.0075 |
| milo_LW_weight_decay_0.001 | 0.9999 | 0.0001 | 0.0001 | 0.9996 | 1.0002 |
| milo_LW_weight_decay_0.0 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

#### Pairwise Significance Tests

| Optimizer A                | Optimizer B               |   Mean A |   Mean B | Better                    |   p-value | Significant   | Metric               |
|:---------------------------|:--------------------------|---------:|---------:|:--------------------------|----------:|:--------------|:---------------------|
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.001 | 0.995479 | 0.999882 | milo_LW_weight_decay_0.001 |  0.254747 |               | final_validation_auc |
| milo_LW_weight_decay_0.0001 | milo_LW_weight_decay_0.0   | 0.995479 | 0.999991 | milo_LW_weight_decay_0.0   |  0.246785 |               | final_validation_auc |
| milo_LW_weight_decay_0.001  | milo_LW_weight_decay_0.0   | 0.999882 | 0.999991 | milo_LW_weight_decay_0.0   |  0.225526 |               | final_validation_auc |

