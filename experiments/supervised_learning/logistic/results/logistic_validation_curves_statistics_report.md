# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.3508 | 0.0041 | 0.0018 | 0.3457 | 0.3559 |
| MILO_LW | 0.3040 | 0.0031 | 0.0014 | 0.3001 | 0.3079 |
| SGD | 0.3067 | 0.0014 | 0.0006 | 0.3050 | 0.3084 |
| ADAMW | 0.2877 | 0.0012 | 0.0005 | 0.2861 | 0.2892 |
| ADAGRAD | 0.4810 | 0.0028 | 0.0012 | 0.4776 | 0.4845 |
| NOVOGRAD | 0.3092 | 0.0014 | 0.0006 | 0.3074 | 0.3110 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.350833 | 0.304008 | MILO_LW  | 8.13499e-08 | ***           | final_validation_loss |
| MILO          | SGD           | 0.350833 | 0.306714 | SGD      | 3.62048e-06 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.350833 | 0.287661 | ADAMW    | 9.3198e-07  | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.350833 | 0.48101  | MILO     | 9.9014e-11  | ***           | final_validation_loss |
| MILO          | NOVOGRAD      | 0.350833 | 0.309194 | NOVOGRAD | 4.2378e-06  | ***           | final_validation_loss |
| MILO_LW       | SGD           | 0.304008 | 0.306714 | MILO_LW  | 0.132372    |               | final_validation_loss |
| MILO_LW       | ADAMW         | 0.304008 | 0.287661 | ADAMW    | 9.06983e-05 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.304008 | 0.48101  | MILO_LW  | 2.49227e-13 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.304008 | 0.309194 | MILO_LW  | 0.0167816   | *             | final_validation_loss |
| SGD           | ADAMW         | 0.306714 | 0.287661 | ADAMW    | 1.66218e-08 | ***           | final_validation_loss |
| SGD           | ADAGRAD       | 0.306714 | 0.48101  | SGD      | 2.60849e-11 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 0.306714 | 0.309194 | SGD      | 0.0246024   | *             | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.287661 | 0.48101  | ADAMW    | 4.96532e-11 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.287661 | 0.309194 | ADAMW    | 9.13035e-09 | ***           | final_validation_loss |
| ADAGRAD       | NOVOGRAD      | 0.48101  | 0.309194 | NOVOGRAD | 1.79728e-11 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 91.3556 | 0.1299 | 0.0581 | 91.1943 | 91.5168 |
| MILO_LW | 91.7556 | 0.1430 | 0.0640 | 91.5780 | 91.9331 |
| SGD | 91.4815 | 0.1610 | 0.0720 | 91.2816 | 91.6813 |
| ADAMW | 92.0247 | 0.0776 | 0.0347 | 91.9283 | 92.1210 |
| ADAGRAD | 87.7457 | 0.1649 | 0.0737 | 87.5410 | 87.9504 |
| NOVOGRAD | 91.3877 | 0.0651 | 0.0291 | 91.3068 | 91.4685 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  91.3556 |  91.7556 | MILO_LW  | 0.00173043  | **            | final_validation_accuracy |
| MILO          | SGD           |  91.3556 |  91.4815 | SGD      | 0.212108    |               | final_validation_accuracy |
| MILO          | ADAMW         |  91.3556 |  92.0247 | ADAMW    | 3.61735e-05 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  91.3556 |  87.7457 | MILO     | 5.68921e-10 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  91.3556 |  91.3877 | NOVOGRAD | 0.639181    |               | final_validation_accuracy |
| MILO_LW       | SGD           |  91.7556 |  91.4815 | MILO_LW  | 0.021912    | *             | final_validation_accuracy |
| MILO_LW       | ADAMW         |  91.7556 |  92.0247 | ADAMW    | 0.00961904  | **            | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  91.7556 |  87.7457 | MILO_LW  | 1.92609e-10 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  91.7556 |  91.3877 | MILO_LW  | 0.00241621  | **            | final_validation_accuracy |
| SGD           | ADAMW         |  91.4815 |  92.0247 | ADAMW    | 0.000587939 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  91.4815 |  87.7457 | SGD      | 3.70582e-10 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  91.4815 |  91.3877 | SGD      | 0.278292    |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  92.0247 |  87.7457 | ADAMW    | 7.3076e-09  | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  92.0247 |  91.3877 | ADAMW    | 8.41852e-07 | ***           | final_validation_accuracy |
| ADAGRAD       | NOVOGRAD      |  87.7457 |  91.3877 | NOVOGRAD | 5.23176e-08 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9129 | 0.0014 | 0.0006 | 0.9111 | 0.9146 |
| MILO_LW | 0.9170 | 0.0014 | 0.0006 | 0.9152 | 0.9187 |
| SGD | 0.9140 | 0.0016 | 0.0007 | 0.9121 | 0.9160 |
| ADAMW | 0.9196 | 0.0008 | 0.0004 | 0.9186 | 0.9206 |
| ADAGRAD | 0.8764 | 0.0017 | 0.0008 | 0.8743 | 0.8785 |
| NOVOGRAD | 0.9131 | 0.0006 | 0.0003 | 0.9123 | 0.9138 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.912861 | 0.916955 | MILO_LW  | 0.00175279  | **            | final_validation_f1_score |
| MILO          | SGD           | 0.912861 | 0.914048 | SGD      | 0.241945    |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.912861 | 0.919612 | ADAMW    | 5.82021e-05 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.912861 | 0.876378 | MILO     | 5.42161e-10 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.912861 | 0.913075 | NOVOGRAD | 0.763541    |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.916955 | 0.914048 | MILO_LW  | 0.0156224   | *             | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.916955 | 0.919612 | ADAMW    | 0.00994675  | **            | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.916955 | 0.876378 | MILO_LW  | 2.24596e-10 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.916955 | 0.913075 | MILO_LW  | 0.00193199  | **            | final_validation_f1_score |
| SGD           | ADAMW         | 0.914048 | 0.919612 | ADAMW    | 0.000433298 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.914048 | 0.876378 | SGD      | 3.79011e-10 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.914048 | 0.913075 | SGD      | 0.251483    |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.919612 | 0.876378 | ADAMW    | 8.47333e-09 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.919612 | 0.913075 | ADAMW    | 8.29176e-07 | ***           | final_validation_f1_score |
| ADAGRAD       | NOVOGRAD      | 0.876378 | 0.913075 | NOVOGRAD | 9.06177e-08 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9921 | 0.0001 | 0.0001 | 0.9920 | 0.9922 |
| MILO_LW | 0.9933 | 0.0001 | 0.0000 | 0.9932 | 0.9934 |
| SGD | 0.9926 | 0.0001 | 0.0000 | 0.9925 | 0.9928 |
| ADAMW | 0.9934 | 0.0001 | 0.0000 | 0.9933 | 0.9934 |
| ADAGRAD | 0.9882 | 0.0002 | 0.0001 | 0.9880 | 0.9884 |
| NOVOGRAD | 0.9926 | 0.0001 | 0.0000 | 0.9925 | 0.9926 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.9921   | 0.99327  | MILO_LW  | 2.15506e-07 | ***           | final_validation_auc |
| MILO          | SGD           | 0.9921   | 0.992642 | SGD      | 4.68629e-05 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.9921   | 0.993354 | ADAMW    | 6.8042e-07  | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.9921   | 0.98818  | MILO     | 2.83454e-10 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.9921   | 0.992553 | NOVOGRAD | 0.000159179 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.99327  | 0.992642 | MILO_LW  | 5.53384e-06 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.99327  | 0.993354 | ADAMW    | 0.0993208   |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.99327  | 0.98818  | MILO_LW  | 7.97312e-10 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.99327  | 0.992553 | MILO_LW  | 5.00373e-07 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.992642 | 0.993354 | ADAMW    | 4.82897e-06 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.992642 | 0.98818  | SGD      | 3.31983e-10 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.992642 | 0.992553 | SGD      | 0.146772    |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.993354 | 0.98818  | ADAMW    | 9.08621e-09 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.993354 | 0.992553 | ADAMW    | 9.28924e-08 | ***           | final_validation_auc |
| ADAGRAD       | NOVOGRAD      | 0.98818  | 0.992553 | NOVOGRAD | 4.94637e-09 | ***           | final_validation_auc |

