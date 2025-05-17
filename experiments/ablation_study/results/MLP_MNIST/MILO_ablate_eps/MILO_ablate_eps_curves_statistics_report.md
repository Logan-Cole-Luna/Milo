# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_eps_1e-05 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| MILO_eps_0.005 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0002 |
| MILO_eps_0.05 | 0.0002 | 0.0000 | 0.0000 | -0.0000 | 0.0003 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A    | Optimizer B    |      Mean A |      Mean B | Better         |   p-value | Significant   | Metric                |
|:---------------|:---------------|------------:|------------:|:---------------|----------:|:--------------|:----------------------|
| MILO_eps_1e-05 | MILO_eps_0.005 | 6.63751e-05 | 8.39256e-05 | MILO_eps_1e-05 | 0.188641  |               | final_validation_loss |
| MILO_eps_1e-05 | MILO_eps_0.05  | 6.63751e-05 | 0.00016343  | MILO_eps_1e-05 | 0.0796909 |               | final_validation_loss |
| MILO_eps_0.005 | MILO_eps_0.05  | 8.39256e-05 | 0.00016343  | MILO_eps_0.005 | 0.0662713 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_eps_1e-05 | 99.9975 | 0.0012 | 0.0008 | 99.9869 | 100.0081 |
| MILO_eps_0.005 | 99.9975 | 0.0012 | 0.0008 | 99.9869 | 100.0081 |
| MILO_eps_0.05 | 99.9958 | 0.0012 | 0.0008 | 99.9852 | 100.0064 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A    | Optimizer B    |   Mean A |   Mean B | Better         |   p-value | Significant   | Metric                    |
|:---------------|:---------------|---------:|---------:|:---------------|----------:|:--------------|:--------------------------|
| MILO_eps_1e-05 | MILO_eps_0.005 |  99.9975 |  99.9975 | MILO_eps_0.005 |  1        |               | final_validation_accuracy |
| MILO_eps_1e-05 | MILO_eps_0.05  |  99.9975 |  99.9958 | MILO_eps_1e-05 |  0.292893 |               | final_validation_accuracy |
| MILO_eps_0.005 | MILO_eps_0.05  |  99.9975 |  99.9958 | MILO_eps_0.005 |  0.292893 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_eps_1e-05 | 1.0000 | 0.0000 | 0.0000 | 0.9999 | 1.0001 |
| MILO_eps_0.005 | 1.0000 | 0.0000 | 0.0000 | 0.9999 | 1.0001 |
| MILO_eps_0.05 | 1.0000 | 0.0000 | 0.0000 | 0.9999 | 1.0001 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A    | Optimizer B    |   Mean A |   Mean B | Better         |   p-value | Significant   | Metric                    |
|:---------------|:---------------|---------:|---------:|:---------------|----------:|:--------------|:--------------------------|
| MILO_eps_1e-05 | MILO_eps_0.005 | 0.999974 | 0.999974 | MILO_eps_0.005 |  1        |               | final_validation_f1_score |
| MILO_eps_1e-05 | MILO_eps_0.05  | 0.999974 | 0.999958 | MILO_eps_1e-05 |  0.311742 |               | final_validation_f1_score |
| MILO_eps_0.005 | MILO_eps_0.05  | 0.999974 | 0.999958 | MILO_eps_0.005 |  0.311742 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_eps_1e-05 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| MILO_eps_0.005 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| MILO_eps_0.05 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A    | Optimizer B    |   Mean A |   Mean B | Better         |   p-value | Significant   | Metric               |
|:---------------|:---------------|---------:|---------:|:---------------|----------:|:--------------|:---------------------|
| MILO_eps_1e-05 | MILO_eps_0.005 |        1 |        1 | MILO_eps_0.005 |     nan   |               | final_validation_auc |
| MILO_eps_1e-05 | MILO_eps_0.05  |        1 |        1 | MILO_eps_1e-05 |       0.5 |               | final_validation_auc |
| MILO_eps_0.005 | MILO_eps_0.05  |        1 |        1 | MILO_eps_0.005 |       0.5 |               | final_validation_auc |

