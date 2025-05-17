# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_eps_1e-05 | 0.0302 | 0.0039 | 0.0028 | -0.0050 | 0.0655 |
| MILO_eps_0.005 | 0.0243 | 0.0021 | 0.0015 | 0.0057 | 0.0428 |
| MILO_eps_0.05 | 0.0437 | 0.0042 | 0.0030 | 0.0056 | 0.0818 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A    | Optimizer B    |    Mean A |    Mean B | Better         |   p-value | Significant   | Metric                |
|:---------------|:---------------|----------:|----------:|:---------------|----------:|:--------------|:----------------------|
| MILO_eps_1e-05 | MILO_eps_0.005 | 0.0302108 | 0.024299  | MILO_eps_0.005 | 0.238866  |               | final_validation_loss |
| MILO_eps_1e-05 | MILO_eps_0.05  | 0.0302108 | 0.0436848 | MILO_eps_1e-05 | 0.0815123 |               | final_validation_loss |
| MILO_eps_0.005 | MILO_eps_0.05  | 0.024299  | 0.0436848 | MILO_eps_0.005 | 0.056222  |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_eps_1e-05 | 98.9090 | 0.1372 | 0.0970 | 97.6765 | 100.1415 |
| MILO_eps_0.005 | 99.1250 | 0.0693 | 0.0490 | 98.5024 | 99.7476 |
| MILO_eps_0.05 | 98.4980 | 0.1188 | 0.0840 | 97.4307 | 99.5653 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A    | Optimizer B    |   Mean A |   Mean B | Better         |   p-value | Significant   | Metric                    |
|:---------------|:---------------|---------:|---------:|:---------------|----------:|:--------------|:--------------------------|
| MILO_eps_1e-05 | MILO_eps_0.005 |   98.909 |   99.125 | MILO_eps_0.005 |  0.228047 |               | final_validation_accuracy |
| MILO_eps_1e-05 | MILO_eps_0.05  |   98.909 |   98.498 | MILO_eps_1e-05 |  0.087466 |               | final_validation_accuracy |
| MILO_eps_0.005 | MILO_eps_0.05  |   99.125 |   98.498 | MILO_eps_0.005 |  0.038737 | *             | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_eps_1e-05 | 0.9891 | 0.0014 | 0.0010 | 0.9767 | 1.0015 |
| MILO_eps_0.005 | 0.9912 | 0.0007 | 0.0005 | 0.9850 | 0.9975 |
| MILO_eps_0.05 | 0.9850 | 0.0013 | 0.0009 | 0.9736 | 0.9964 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A    | Optimizer B    |   Mean A |   Mean B | Better         |   p-value | Significant   | Metric                    |
|:---------------|:---------------|---------:|---------:|:---------------|----------:|:--------------|:--------------------------|
| MILO_eps_1e-05 | MILO_eps_0.005 | 0.989098 | 0.991246 | MILO_eps_0.005 | 0.230253  |               | final_validation_f1_score |
| MILO_eps_1e-05 | MILO_eps_0.05  | 0.989098 | 0.984981 | MILO_eps_1e-05 | 0.090387  |               | final_validation_f1_score |
| MILO_eps_0.005 | MILO_eps_0.05  | 0.991246 | 0.984981 | MILO_eps_0.005 | 0.0457402 | *             | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_eps_1e-05 | 0.9999 | 0.0000 | 0.0000 | 0.9997 | 1.0001 |
| MILO_eps_0.005 | 1.0000 | 0.0000 | 0.0000 | 0.9999 | 1.0000 |
| MILO_eps_0.05 | 0.9999 | 0.0000 | 0.0000 | 0.9999 | 0.9999 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A    | Optimizer B    |   Mean A |   Mean B | Better         |   p-value | Significant   | Metric               |
|:---------------|:---------------|---------:|---------:|:---------------|----------:|:--------------|:---------------------|
| MILO_eps_1e-05 | MILO_eps_0.005 | 0.999911 | 0.99995  | MILO_eps_0.005 | 0.272267  |               | final_validation_auc |
| MILO_eps_1e-05 | MILO_eps_0.05  | 0.999911 | 0.999905 | MILO_eps_1e-05 | 0.78914   |               | final_validation_auc |
| MILO_eps_0.005 | MILO_eps_0.05  | 0.99995  | 0.999905 | MILO_eps_0.005 | 0.0473264 | *             | final_validation_auc |

