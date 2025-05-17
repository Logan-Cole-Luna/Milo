# Statistical Analysis Report

## Methodology

Standard deviations, standard errors, and 95% confidence intervals were computed using the t-distribution.

### Validation_Loss Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_eps_1e-05 | 0.0060 | 0.0014 | 0.0008 | 0.0025 | 0.0094 |
| milo_LW_eps_0.005 | 0.0053 | 0.0008 | 0.0005 | 0.0033 | 0.0074 |
| milo_LW_eps_0.05 | 0.0180 | 0.0073 | 0.0042 | -0.0001 | 0.0361 |

#### Pairwise Significance Tests

| Optimizer A      | Optimizer B      |     Mean A |     Mean B | Better           |   p-value | Significant   | Metric                |
|:-----------------|:-----------------|-----------:|-----------:|:-----------------|----------:|:--------------|:----------------------|
| milo_LW_eps_1e-05 | milo_LW_eps_0.005 | 0.00596096 | 0.00532276 | milo_LW_eps_0.005 | 0.540841  |               | final_validation_loss |
| milo_LW_eps_1e-05 | milo_LW_eps_0.05  | 0.00596096 | 0.0180036  | milo_LW_eps_1e-05 | 0.0982216 |               | final_validation_loss |
| milo_LW_eps_0.005 | milo_LW_eps_0.05  | 0.00532276 | 0.0180036  | milo_LW_eps_0.005 | 0.0926192 |               | final_validation_loss |

### Validation_Accuracy Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_eps_1e-05 | 99.7913 | 0.0400 | 0.0231 | 99.6919 | 99.8907 |
| milo_LW_eps_0.005 | 99.8227 | 0.0283 | 0.0163 | 99.7523 | 99.8930 |
| milo_LW_eps_0.05 | 99.3773 | 0.2645 | 0.1527 | 98.7203 | 100.0344 |

#### Pairwise Significance Tests

| Optimizer A      | Optimizer B      |   Mean A |   Mean B | Better           |   p-value | Significant   | Metric                    |
|:-----------------|:-----------------|---------:|---------:|:-----------------|----------:|:--------------|:--------------------------|
| milo_LW_eps_1e-05 | milo_LW_eps_0.005 |  99.7913 |  99.8227 | milo_LW_eps_0.005 | 0.33664   |               | final_validation_accuracy |
| milo_LW_eps_1e-05 | milo_LW_eps_0.05  |  99.7913 |  99.3773 | milo_LW_eps_1e-05 | 0.110238  |               | final_validation_accuracy |
| milo_LW_eps_0.005 | milo_LW_eps_0.05  |  99.8227 |  99.3773 | milo_LW_eps_0.005 | 0.0985762 |               | final_validation_accuracy |

### Validation_F1_Score Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_eps_1e-05 | 0.9979 | 0.0004 | 0.0002 | 0.9969 | 0.9989 |
| milo_LW_eps_0.005 | 0.9982 | 0.0003 | 0.0002 | 0.9975 | 0.9989 |
| milo_LW_eps_0.05 | 0.9938 | 0.0026 | 0.0015 | 0.9872 | 1.0003 |

#### Pairwise Significance Tests

| Optimizer A      | Optimizer B      |   Mean A |   Mean B | Better           |   p-value | Significant   | Metric                    |
|:-----------------|:-----------------|---------:|---------:|:-----------------|----------:|:--------------|:--------------------------|
| milo_LW_eps_1e-05 | milo_LW_eps_0.005 | 0.997914 | 0.998227 | milo_LW_eps_0.005 | 0.336766  |               | final_validation_f1_score |
| milo_LW_eps_1e-05 | milo_LW_eps_0.05  | 0.997914 | 0.99377  | milo_LW_eps_1e-05 | 0.109998  |               | final_validation_f1_score |
| milo_LW_eps_0.005 | milo_LW_eps_0.05  | 0.998227 | 0.99377  | milo_LW_eps_0.005 | 0.0983756 |               | final_validation_f1_score |

### Validation_Auc Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_eps_1e-05 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| milo_LW_eps_0.005 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| milo_LW_eps_0.05 | 1.0000 | 0.0000 | 0.0000 | 0.9999 | 1.0000 |

#### Pairwise Significance Tests

| Optimizer A      | Optimizer B      |   Mean A |   Mean B | Better           |   p-value | Significant   | Metric               |
|:-----------------|:-----------------|---------:|---------:|:-----------------|----------:|:--------------|:---------------------|
| milo_LW_eps_1e-05 | milo_LW_eps_0.005 | 0.999997 | 0.999998 | milo_LW_eps_0.005 |  0.370918 |               | final_validation_auc |
| milo_LW_eps_1e-05 | milo_LW_eps_0.05  | 0.999997 | 0.999979 | milo_LW_eps_1e-05 |  0.210172 |               | final_validation_auc |
| milo_LW_eps_0.005 | milo_LW_eps_0.05  | 0.999998 | 0.999979 | milo_LW_eps_0.005 |  0.195965 |               | final_validation_auc |

