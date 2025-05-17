# Statistical Analysis Report

## Methodology

Standard deviations, standard errors, and 95% confidence intervals were computed using the t-distribution.

### Validation_Loss Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_momentum_0.9 | 0.5341 | 0.0329 | 0.0190 | 0.4524 | 0.6158 |
| milo_LW_momentum_0.0 | 0.5317 | 0.0351 | 0.0203 | 0.4445 | 0.6190 |
| milo_LW_momentum_0.45 | 0.5169 | 0.0326 | 0.0188 | 0.4361 | 0.5978 |

#### Pairwise Significance Tests

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric                |
|:--------------------|:---------------------|---------:|---------:|:---------------------|----------:|:--------------|:----------------------|
| milo_LW_momentum_0.9 | milo_LW_momentum_0.0  | 0.534107 | 0.531729 | milo_LW_momentum_0.0  |  0.935911 |               | final_validation_loss |
| milo_LW_momentum_0.9 | milo_LW_momentum_0.45 | 0.534107 | 0.516928 | milo_LW_momentum_0.45 |  0.55525  |               | final_validation_loss |
| milo_LW_momentum_0.0 | milo_LW_momentum_0.45 | 0.531729 | 0.516928 | milo_LW_momentum_0.45 |  0.620933 |               | final_validation_loss |

### Validation_Accuracy Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_momentum_0.9 | 81.8267 | 1.1894 | 0.6867 | 78.8721 | 84.7813 |
| milo_LW_momentum_0.0 | 81.7033 | 1.1000 | 0.6351 | 78.9707 | 84.4359 |
| milo_LW_momentum_0.45 | 82.4467 | 1.0498 | 0.6061 | 79.8389 | 85.0545 |

#### Pairwise Significance Tests

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric                    |
|:--------------------|:---------------------|---------:|---------:|:---------------------|----------:|:--------------|:--------------------------|
| milo_LW_momentum_0.9 | milo_LW_momentum_0.0  |  81.8267 |  81.7033 | milo_LW_momentum_0.9  |  0.9015   |               | final_validation_accuracy |
| milo_LW_momentum_0.9 | milo_LW_momentum_0.45 |  81.8267 |  82.4467 | milo_LW_momentum_0.45 |  0.53612  |               | final_validation_accuracy |
| milo_LW_momentum_0.0 | milo_LW_momentum_0.45 |  81.7033 |  82.4467 | milo_LW_momentum_0.45 |  0.444929 |               | final_validation_accuracy |

### Validation_F1_Score Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_momentum_0.9 | 0.8179 | 0.0118 | 0.0068 | 0.7885 | 0.8473 |
| milo_LW_momentum_0.0 | 0.8178 | 0.0102 | 0.0059 | 0.7924 | 0.8432 |
| milo_LW_momentum_0.45 | 0.8248 | 0.0106 | 0.0061 | 0.7985 | 0.8512 |

#### Pairwise Significance Tests

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric                    |
|:--------------------|:---------------------|---------:|---------:|:---------------------|----------:|:--------------|:--------------------------|
| milo_LW_momentum_0.9 | milo_LW_momentum_0.0  | 0.817886 | 0.817795 | milo_LW_momentum_0.9  |  0.992413 |               | final_validation_f1_score |
| milo_LW_momentum_0.9 | milo_LW_momentum_0.45 | 0.817886 | 0.824811 | milo_LW_momentum_0.45 |  0.49299  |               | final_validation_f1_score |
| milo_LW_momentum_0.0 | milo_LW_momentum_0.45 | 0.817795 | 0.824811 | milo_LW_momentum_0.45 |  0.455588 |               | final_validation_f1_score |

### Validation_Auc Statistics

Number of runs: 3

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| milo_LW_momentum_0.9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_LW_momentum_0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| milo_LW_momentum_0.45 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Pairwise Significance Tests

| Optimizer A         | Optimizer B          |   Mean A |   Mean B | Better               |   p-value | Significant   | Metric               |
|:--------------------|:---------------------|---------:|---------:|:---------------------|----------:|:--------------|:---------------------|
| milo_LW_momentum_0.9 | milo_LW_momentum_0.0  |        0 |        0 | milo_LW_momentum_0.0  |       nan |               | final_validation_auc |
| milo_LW_momentum_0.9 | milo_LW_momentum_0.45 |        0 |        0 | milo_LW_momentum_0.45 |       nan |               | final_validation_auc |
| milo_LW_momentum_0.0 | milo_LW_momentum_0.45 |        0 |        0 | milo_LW_momentum_0.45 |       nan |               | final_validation_auc |

