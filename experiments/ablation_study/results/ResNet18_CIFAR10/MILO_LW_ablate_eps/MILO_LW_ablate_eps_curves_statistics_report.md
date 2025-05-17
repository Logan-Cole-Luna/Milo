# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_eps_1e-05 | 0.3581 | 0.0293 | 0.0207 | 0.0949 | 0.6213 |
| MILO_LW_eps_0.005 | 0.4478 | 0.0599 | 0.0424 | -0.0904 | 0.9859 |
| MILO_LW_eps_0.05 | 3.3784 | 4.0958 | 2.8962 | -33.4213 | 40.1780 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric                |
|:------------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:----------------------|
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.005 | 0.358098 | 0.447797 | MILO_LW_eps_1e-05 |  0.242893 |               | final_validation_loss |
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.05  | 0.358098 | 3.37836  | MILO_LW_eps_1e-05 |  0.486646 |               | final_validation_loss |
| MILO_LW_eps_0.005 | MILO_LW_eps_0.05  | 0.447797 | 3.37836  | MILO_LW_eps_0.005 |  0.496222 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_eps_1e-05 | 95.8570 | 0.1626 | 0.1150 | 94.3958 | 97.3182 |
| MILO_LW_eps_0.005 | 95.5490 | 0.2277 | 0.1610 | 93.5033 | 97.5947 |
| MILO_LW_eps_0.05 | 88.2680 | 10.3577 | 7.3240 | -4.7922 | 181.3282 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric                    |
|:------------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:--------------------------|
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.005 |   95.857 |   95.549 | MILO_LW_eps_1e-05 |  0.272325 |               | final_validation_accuracy |
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.05  |   95.857 |   88.268 | MILO_LW_eps_1e-05 |  0.488661 |               | final_validation_accuracy |
| MILO_LW_eps_0.005 | MILO_LW_eps_0.05  |   95.549 |   88.268 | MILO_LW_eps_0.005 |  0.501824 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_eps_1e-05 | 0.9587 | 0.0019 | 0.0013 | 0.9416 | 0.9757 |
| MILO_LW_eps_0.005 | 0.9557 | 0.0021 | 0.0015 | 0.9372 | 0.9742 |
| MILO_LW_eps_0.05 | 0.8844 | 0.1015 | 0.0718 | -0.0275 | 1.7963 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric                    |
|:------------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:--------------------------|
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.005 | 0.958683 | 0.955686 | MILO_LW_eps_1e-05 |  0.270184 |               | final_validation_f1_score |
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.05  | 0.958683 | 0.884396 | MILO_LW_eps_1e-05 |  0.488969 |               | final_validation_f1_score |
| MILO_LW_eps_0.005 | MILO_LW_eps_0.05  | 0.955686 | 0.884396 | MILO_LW_eps_0.005 |  0.502072 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_LW_eps_1e-05 | 0.9986 | 0.0002 | 0.0001 | 0.9972 | 1.0000 |
| MILO_LW_eps_0.005 | 0.9986 | 0.0001 | 0.0000 | 0.9980 | 0.9991 |
| MILO_LW_eps_0.05 | 0.9923 | 0.0087 | 0.0062 | 0.9139 | 1.0707 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A       | Optimizer B       |   Mean A |   Mean B | Better            |   p-value | Significant   | Metric               |
|:------------------|:------------------|---------:|---------:|:------------------|----------:|:--------------|:---------------------|
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.005 | 0.998618 | 0.998551 | MILO_LW_eps_1e-05 |  0.658395 |               | final_validation_auc |
| MILO_LW_eps_1e-05 | MILO_LW_eps_0.05  | 0.998618 | 0.992294 | MILO_LW_eps_1e-05 |  0.492017 |               | final_validation_auc |
| MILO_LW_eps_0.005 | MILO_LW_eps_0.05  | 0.998551 | 0.992294 | MILO_LW_eps_0.005 |  0.49546  |               | final_validation_auc |

