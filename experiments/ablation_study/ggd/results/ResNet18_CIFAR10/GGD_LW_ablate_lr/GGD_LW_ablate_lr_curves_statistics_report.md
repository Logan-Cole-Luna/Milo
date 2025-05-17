# Statistical Analysis Report

## Methodology

Standard deviations, standard errors, and 95% confidence intervals were computed using the t-distribution.

### Validation_Loss Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_lr_0.01 | 0.5824 | 0.0168 | 0.0097 | 0.5407 | 0.6242 |
| milo_LW_lr_0.1 | 0.6384 | 0.0572 | 0.0330 | 0.4962 | 0.7806 |
| milo_LW_lr_0.001 | 0.4693 | 0.0109 | 0.0063 | 0.4421 | 0.4965 |

#### Pairwise Significance Tests

| Optimizer A    | Optimizer B     |   Mean A |   Mean B | Better          |    p-value | Significant   | Metric                |
|:---------------|:----------------|---------:|---------:|:----------------|-----------:|:--------------|:----------------------|
| milo_LW_lr_0.01 | milo_LW_lr_0.1   | 0.582441 | 0.638394 | milo_LW_lr_0.01  | 0.227426   |               | final_validation_loss |
| milo_LW_lr_0.01 | milo_LW_lr_0.001 | 0.582441 | 0.469285 | milo_LW_lr_0.001 | 0.00125672 | **            | final_validation_loss |
| milo_LW_lr_0.1  | milo_LW_lr_0.001 | 0.638394 | 0.469285 | milo_LW_lr_0.001 | 0.0322918  | *             | final_validation_loss |

### Validation_Accuracy Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_lr_0.01 | 80.0633 | 0.5352 | 0.3090 | 78.7338 | 81.3928 |
| milo_LW_lr_0.1 | 78.5167 | 1.5382 | 0.8881 | 74.6956 | 82.3377 |
| milo_LW_lr_0.001 | 84.2433 | 0.1607 | 0.0928 | 83.8441 | 84.6426 |

#### Pairwise Significance Tests

| Optimizer A    | Optimizer B     |   Mean A |   Mean B | Better          |    p-value | Significant   | Metric                    |
|:---------------|:----------------|---------:|---------:|:----------------|-----------:|:--------------|:--------------------------|
| milo_LW_lr_0.01 | milo_LW_lr_0.1   |  80.0633 |  78.5167 | milo_LW_lr_0.01  | 0.217294   |               | final_validation_accuracy |
| milo_LW_lr_0.01 | milo_LW_lr_0.001 |  80.0633 |  84.2433 | milo_LW_lr_0.001 | 0.00302987 | **            | final_validation_accuracy |
| milo_LW_lr_0.1  | milo_LW_lr_0.001 |  78.5167 |  84.2433 | milo_LW_lr_0.001 | 0.0222261  | *             | final_validation_accuracy |

### Validation_F1_Score Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_lr_0.01 | 0.8002 | 0.0060 | 0.0035 | 0.7853 | 0.8151 |
| milo_LW_lr_0.1 | 0.7793 | 0.0193 | 0.0111 | 0.7314 | 0.8272 |
| milo_LW_lr_0.001 | 0.8424 | 0.0024 | 0.0014 | 0.8365 | 0.8483 |

#### Pairwise Significance Tests

| Optimizer A    | Optimizer B     |   Mean A |   Mean B | Better          |    p-value | Significant   | Metric                    |
|:---------------|:----------------|---------:|---------:|:----------------|-----------:|:--------------|:--------------------------|
| milo_LW_lr_0.01 | milo_LW_lr_0.1   | 0.800173 | 0.779307 | milo_LW_lr_0.01  | 0.194895   |               | final_validation_f1_score |
| milo_LW_lr_0.01 | milo_LW_lr_0.001 | 0.800173 | 0.84237  | milo_LW_lr_0.001 | 0.00273303 | **            | final_validation_f1_score |
| milo_LW_lr_0.1  | milo_LW_lr_0.001 | 0.779307 | 0.84237  | milo_LW_lr_0.001 | 0.0282782  | *             | final_validation_f1_score |

### Validation_Auc Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_lr_0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_LW_lr_0.1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_LW_lr_0.001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Pairwise Significance Tests

| Optimizer A    | Optimizer B     |   Mean A |   Mean B | Better          |   p-value | Significant   | Metric               |
|:---------------|:----------------|---------:|---------:|:----------------|----------:|:--------------|:---------------------|
| milo_LW_lr_0.01 | milo_LW_lr_0.1   |        0 |        0 | milo_LW_lr_0.1   |       nan |               | final_validation_auc |
| milo_LW_lr_0.01 | milo_LW_lr_0.001 |        0 |        0 | milo_LW_lr_0.001 |       nan |               | final_validation_auc |
| milo_LW_lr_0.1  | milo_LW_lr_0.001 |        0 |        0 | milo_LW_lr_0.001 |       nan |               | final_validation_auc |

