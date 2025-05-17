# Statistical Analysis Report

## Methodology

Standard deviations, standard errors, and 95% confidence intervals were computed using the t-distribution.

### Validation_Loss Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_eps_1e-05 | 0.4369 | 0.0123 | 0.0071 | 0.4062 | 0.4675 |
| milo_eps_0.005 | 0.4088 | 0.0194 | 0.0112 | 0.3607 | 0.4570 |
| milo_eps_0.05 | 0.4047 | 0.0203 | 0.0117 | 0.3542 | 0.4552 |

#### Pairwise Significance Tests

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |   p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:--------------|----------:|:--------------|:----------------------|
| milo_eps_1e-05 | milo_eps_0.005 | 0.436851 | 0.408839 | milo_eps_0.005 | 0.114641  |               | final_validation_loss |
| milo_eps_1e-05 | milo_eps_0.05  | 0.436851 | 0.40471  | milo_eps_0.05  | 0.0933086 |               | final_validation_loss |
| milo_eps_0.005 | milo_eps_0.05  | 0.408839 | 0.40471  | milo_eps_0.05  | 0.811644  |               | final_validation_loss |

### Validation_Accuracy Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_eps_1e-05 | 85.4600 | 0.4233 | 0.2444 | 84.4084 | 86.5116 |
| milo_eps_0.005 | 86.1333 | 0.4917 | 0.2839 | 84.9120 | 87.3547 |
| milo_eps_0.05 | 86.4100 | 0.6393 | 0.3691 | 84.8219 | 87.9981 |

#### Pairwise Significance Tests

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:--------------|----------:|:--------------|:--------------------------|
| milo_eps_1e-05 | milo_eps_0.005 |  85.46   |  86.1333 | milo_eps_0.005 |  0.148234 |               | final_validation_accuracy |
| milo_eps_1e-05 | milo_eps_0.05  |  85.46   |  86.41   | milo_eps_0.05  |  0.108861 |               | final_validation_accuracy |
| milo_eps_0.005 | milo_eps_0.05  |  86.1333 |  86.41   | milo_eps_0.05  |  0.58635  |               | final_validation_accuracy |

### Validation_F1_Score Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_eps_1e-05 | 0.8542 | 0.0042 | 0.0024 | 0.8436 | 0.8647 |
| milo_eps_0.005 | 0.8603 | 0.0056 | 0.0032 | 0.8463 | 0.8742 |
| milo_eps_0.05 | 0.8632 | 0.0069 | 0.0040 | 0.8460 | 0.8804 |

#### Pairwise Significance Tests

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:--------------|----------:|:--------------|:--------------------------|
| milo_eps_1e-05 | milo_eps_0.005 | 0.854165 | 0.860277 | milo_eps_0.005 |  0.21159  |               | final_validation_f1_score |
| milo_eps_1e-05 | milo_eps_0.05  | 0.854165 | 0.863184 | milo_eps_0.05  |  0.141619 |               | final_validation_f1_score |
| milo_eps_0.005 | milo_eps_0.05  | 0.860277 | 0.863184 | milo_eps_0.05  |  0.603799 |               | final_validation_f1_score |

### Validation_Auc Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_eps_1e-05 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_eps_0.005 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_eps_0.05 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Pairwise Significance Tests

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:--------------|----------:|:--------------|:---------------------|
| milo_eps_1e-05 | milo_eps_0.005 |        0 |        0 | milo_eps_0.005 |       nan |               | final_validation_auc |
| milo_eps_1e-05 | milo_eps_0.05  |        0 |        0 | milo_eps_0.05  |       nan |               | final_validation_auc |
| milo_eps_0.005 | milo_eps_0.05  |        0 |        0 | milo_eps_0.05  |       nan |               | final_validation_auc |

