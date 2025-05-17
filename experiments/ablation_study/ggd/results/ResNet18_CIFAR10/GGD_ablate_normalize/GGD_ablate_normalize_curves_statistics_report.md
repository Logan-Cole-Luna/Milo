# Statistical Analysis Report

## Methodology

Standard deviations, standard errors, and 95% confidence intervals were computed using the t-distribution.

### Validation_Loss Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_normalize_True | 0.4502 | 0.0018 | 0.0010 | 0.4457 | 0.4546 |
| milo_normalize_False | 0.4353 | 0.0202 | 0.0117 | 0.3852 | 0.4855 |

#### Pairwise Significance Tests

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better              |   p-value | Significant   | Metric                |
|:-------------------|:--------------------|---------:|---------:|:--------------------|----------:|:--------------|:----------------------|
| milo_normalize_True | milo_normalize_False | 0.450169 | 0.435312 | milo_normalize_False |  0.330243 |               | final_validation_loss |

### Validation_Accuracy Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_normalize_True | 84.7500 | 0.2307 | 0.1332 | 84.1770 | 85.3230 |
| milo_normalize_False | 85.1833 | 0.5829 | 0.3365 | 83.7354 | 86.6313 |

#### Pairwise Significance Tests

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better              |   p-value | Significant   | Metric                    |
|:-------------------|:--------------------|---------:|---------:|:--------------------|----------:|:--------------|:--------------------------|
| milo_normalize_True | milo_normalize_False |    84.75 |  85.1833 | milo_normalize_False |  0.328488 |               | final_validation_accuracy |

### Validation_F1_Score Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_normalize_True | 0.8472 | 0.0020 | 0.0012 | 0.8422 | 0.8521 |
| milo_normalize_False | 0.8521 | 0.0056 | 0.0032 | 0.8382 | 0.8659 |

#### Pairwise Significance Tests

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better              |   p-value | Significant   | Metric                    |
|:-------------------|:--------------------|---------:|---------:|:--------------------|----------:|:--------------|:--------------------------|
| milo_normalize_True | milo_normalize_False | 0.847165 | 0.852082 | milo_normalize_False |  0.262626 |               | final_validation_f1_score |

### Validation_Auc Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_normalize_True | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_normalize_False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Pairwise Significance Tests

| Optimizer A        | Optimizer B         |   Mean A |   Mean B | Better              |   p-value | Significant   | Metric               |
|:-------------------|:--------------------|---------:|---------:|:--------------------|----------:|:--------------|:---------------------|
| milo_normalize_True | milo_normalize_False |        0 |        0 | milo_normalize_False |       nan |               | final_validation_auc |

