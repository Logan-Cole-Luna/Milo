# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_lr_0.01 | 0.0139 | 0.0109 | 0.0077 | -0.0841 | 0.1118 |
| MILO_lr_0.1 | 0.0118 | 0.0057 | 0.0040 | -0.0394 | 0.0630 |
| MILO_lr_0.001 | 0.0032 | 0.0009 | 0.0006 | -0.0047 | 0.0112 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |     Mean B | Better        |   p-value | Significant   | Metric                |
|:--------------|:--------------|----------:|-----------:|:--------------|----------:|:--------------|:----------------------|
| MILO_lr_0.01  | MILO_lr_0.1   | 0.01386   | 0.0117974  | MILO_lr_0.1   |  0.840724 |               | final_validation_loss |
| MILO_lr_0.01  | MILO_lr_0.001 | 0.01386   | 0.00323731 | MILO_lr_0.001 |  0.398447 |               | final_validation_loss |
| MILO_lr_0.1   | MILO_lr_0.001 | 0.0117974 | 0.00323731 | MILO_lr_0.001 |  0.273884 |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_lr_0.01 | 99.6475 | 0.3123 | 0.2208 | 96.8415 | 102.4535 |
| MILO_lr_0.1 | 99.8000 | 0.1108 | 0.0783 | 98.8047 | 100.7953 |
| MILO_lr_0.001 | 99.9450 | 0.0118 | 0.0083 | 99.8391 | 100.0509 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:--------------|----------:|:--------------|:--------------------------|
| MILO_lr_0.01  | MILO_lr_0.1   |  99.6475 |   99.8   | MILO_lr_0.1   |  0.614237 |               | final_validation_accuracy |
| MILO_lr_0.01  | MILO_lr_0.001 |  99.6475 |   99.945 | MILO_lr_0.001 |  0.406266 |               | final_validation_accuracy |
| MILO_lr_0.1   | MILO_lr_0.001 |  99.8    |   99.945 | MILO_lr_0.001 |  0.312612 |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_lr_0.01 | 0.9965 | 0.0031 | 0.0022 | 0.9683 | 1.0246 |
| MILO_lr_0.1 | 0.9980 | 0.0011 | 0.0008 | 0.9881 | 1.0079 |
| MILO_lr_0.001 | 0.9994 | 0.0001 | 0.0001 | 0.9984 | 1.0005 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |   p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:--------------|----------:|:--------------|:--------------------------|
| MILO_lr_0.01  | MILO_lr_0.1   | 0.99648  | 0.997992 | MILO_lr_0.1   |  0.617753 |               | final_validation_f1_score |
| MILO_lr_0.01  | MILO_lr_0.001 | 0.99648  | 0.999444 | MILO_lr_0.001 |  0.408274 |               | final_validation_f1_score |
| MILO_lr_0.1   | MILO_lr_0.001 | 0.997992 | 0.999444 | MILO_lr_0.001 |  0.310314 |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 2

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO_lr_0.01 | 1.0000 | 0.0000 | 0.0000 | 0.9996 | 1.0003 |
| MILO_lr_0.1 | 1.0000 | 0.0000 | 0.0000 | 0.9999 | 1.0001 |
| MILO_lr_0.001 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better        |   p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:--------------|----------:|:--------------|:---------------------|
| MILO_lr_0.01  | MILO_lr_0.1   | 0.999961 | 0.999989 | MILO_lr_0.1   |  0.498768 |               | final_validation_auc |
| MILO_lr_0.01  | MILO_lr_0.001 | 0.999961 | 0.999998 | MILO_lr_0.001 |  0.416609 |               | final_validation_auc |
| MILO_lr_0.1   | MILO_lr_0.001 | 0.999989 | 0.999998 | MILO_lr_0.001 |  0.376227 |               | final_validation_auc |

