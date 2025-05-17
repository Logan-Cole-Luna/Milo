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
| SGD | 0.2973 | 0.0018 | 0.0008 | 0.2952 | 0.2995 |
| ADAMW | 0.5517 | 0.0480 | 0.0215 | 0.4921 | 0.6113 |
| ADAGRAD | 0.2860 | 0.0007 | 0.0003 | 0.2851 | 0.2868 |
| NOVOGRAD | 0.3201 | 0.0111 | 0.0049 | 0.3064 | 0.3339 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.350833 | 0.304008 | MILO_LW  | 8.13499e-08 | ***           | final_validation_loss |
| MILO          | SGD           | 0.350833 | 0.297345 | SGD      | 5.66242e-07 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.350833 | 0.551702 | MILO     | 0.0006857   | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.350833 | 0.285962 | ADAGRAD  | 2.42485e-06 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      | 0.350833 | 0.320129 | NOVOGRAD | 0.00200393  | **            | final_validation_loss |
| MILO_LW       | SGD           | 0.304008 | 0.297345 | SGD      | 0.00545223  | **            | final_validation_loss |
| MILO_LW       | ADAMW         | 0.304008 | 0.551702 | MILO_LW  | 0.00030909  | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.304008 | 0.285962 | ADAGRAD  | 0.000137835 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.304008 | 0.320129 | MILO_LW  | 0.0285412   | *             | final_validation_loss |
| SGD           | ADAMW         | 0.297345 | 0.551702 | SGD      | 0.000286118 | ***           | final_validation_loss |
| SGD           | ADAGRAD       | 0.297345 | 0.285962 | ADAGRAD  | 3.42685e-05 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 0.297345 | 0.320129 | SGD      | 0.00927838  | **            | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.551702 | 0.285962 | ADAGRAD  | 0.000243692 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.551702 | 0.320129 | NOVOGRAD | 0.000269348 | ***           | final_validation_loss |
| ADAGRAD       | NOVOGRAD      | 0.285962 | 0.320129 | ADAGRAD  | 0.0022582   | **            | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 91.3556 | 0.1299 | 0.0581 | 91.1943 | 91.5168 |
| MILO_LW | 91.7556 | 0.1430 | 0.0640 | 91.5780 | 91.9331 |
| SGD | 91.7951 | 0.2395 | 0.1071 | 91.4977 | 92.0924 |
| ADAMW | 88.8667 | 0.9555 | 0.4273 | 87.6803 | 90.0530 |
| ADAGRAD | 92.2148 | 0.1035 | 0.0463 | 92.0863 | 92.3433 |
| NOVOGRAD | 90.9481 | 0.3658 | 0.1636 | 90.4940 | 91.4023 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  91.3556 |  91.7556 | MILO_LW  | 0.00173043  | **            | final_validation_accuracy |
| MILO          | SGD           |  91.3556 |  91.7951 | SGD      | 0.0107526   | *             | final_validation_accuracy |
| MILO          | ADAMW         |  91.3556 |  88.8667 | MILO     | 0.00399914  | **            | final_validation_accuracy |
| MILO          | ADAGRAD       |  91.3556 |  92.2148 | ADAGRAD  | 4.19457e-06 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  91.3556 |  90.9481 | MILO     | 0.0658665   |               | final_validation_accuracy |
| MILO_LW       | SGD           |  91.7556 |  91.7951 | SGD      | 0.761358    |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  91.7556 |  88.8667 | MILO_LW  | 0.00222221  | **            | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  91.7556 |  92.2148 | ADAGRAD  | 0.000562401 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  91.7556 |  90.9481 | MILO_LW  | 0.00532354  | **            | final_validation_accuracy |
| SGD           | ADAMW         |  91.7951 |  88.8667 | SGD      | 0.00173213  | **            | final_validation_accuracy |
| SGD           | ADAGRAD       |  91.7951 |  92.2148 | ADAGRAD  | 0.0134668   | *             | final_validation_accuracy |
| SGD           | NOVOGRAD      |  91.7951 |  90.9481 | SGD      | 0.00355172  | **            | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  88.8667 |  92.2148 | ADAGRAD  | 0.00133099  | **            | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  88.8667 |  90.9481 | NOVOGRAD | 0.00569596  | **            | final_validation_accuracy |
| ADAGRAD       | NOVOGRAD      |  92.2148 |  90.9481 | ADAGRAD  | 0.000948766 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9129 | 0.0014 | 0.0006 | 0.9111 | 0.9146 |
| MILO_LW | 0.9170 | 0.0014 | 0.0006 | 0.9152 | 0.9187 |
| SGD | 0.9173 | 0.0024 | 0.0011 | 0.9143 | 0.9203 |
| ADAMW | 0.8882 | 0.0091 | 0.0041 | 0.8769 | 0.8994 |
| ADAGRAD | 0.9214 | 0.0011 | 0.0005 | 0.9201 | 0.9227 |
| NOVOGRAD | 0.9088 | 0.0036 | 0.0016 | 0.9043 | 0.9132 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.912861 | 0.916955 | MILO_LW  | 0.00175279  | **            | final_validation_f1_score |
| MILO          | SGD           | 0.912861 | 0.917262 | SGD      | 0.0111459   | *             | final_validation_f1_score |
| MILO          | ADAMW         | 0.912861 | 0.888157 | MILO     | 0.00330465  | **            | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.912861 | 0.921422 | ADAGRAD  | 7.19738e-06 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.912861 | 0.908781 | MILO     | 0.0623631   |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.916955 | 0.917262 | SGD      | 0.813962    |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.916955 | 0.888157 | MILO_LW  | 0.00181493  | **            | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.916955 | 0.921422 | ADAGRAD  | 0.000652318 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.916955 | 0.908781 | MILO_LW  | 0.00463202  | **            | final_validation_f1_score |
| SGD           | ADAMW         | 0.917262 | 0.888157 | SGD      | 0.00137507  | **            | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.917262 | 0.921422 | ADAGRAD  | 0.0144226   | *             | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.917262 | 0.908781 | SGD      | 0.00322479  | **            | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.888157 | 0.921422 | ADAGRAD  | 0.00110047  | **            | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.888157 | 0.908781 | NOVOGRAD | 0.00464811  | **            | final_validation_f1_score |
| ADAGRAD       | NOVOGRAD      | 0.921422 | 0.908781 | ADAGRAD  | 0.000851979 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9921 | 0.0001 | 0.0001 | 0.9920 | 0.9922 |
| MILO_LW | 0.9933 | 0.0001 | 0.0000 | 0.9932 | 0.9934 |
| SGD | 0.9929 | 0.0001 | 0.0000 | 0.9928 | 0.9930 |
| ADAMW | 0.9901 | 0.0007 | 0.0003 | 0.9892 | 0.9911 |
| ADAGRAD | 0.9933 | 0.0001 | 0.0000 | 0.9933 | 0.9934 |
| NOVOGRAD | 0.9924 | 0.0003 | 0.0001 | 0.9920 | 0.9927 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.9921   | 0.99327  | MILO_LW  | 2.15506e-07 | ***           | final_validation_auc |
| MILO          | SGD           | 0.9921   | 0.992914 | SGD      | 2.37087e-06 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.9921   | 0.990142 | MILO     | 0.00356901  | **            | final_validation_auc |
| MILO          | ADAGRAD       | 0.9921   | 0.993335 | ADAGRAD  | 7.45757e-07 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.9921   | 0.992362 | NOVOGRAD | 0.0886928   |               | final_validation_auc |
| MILO_LW       | SGD           | 0.99327  | 0.992914 | MILO_LW  | 0.000188016 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.99327  | 0.990142 | MILO_LW  | 0.000616688 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.99327  | 0.993335 | ADAGRAD  | 0.188253    |               | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.99327  | 0.992362 | MILO_LW  | 0.000831991 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.992914 | 0.990142 | SGD      | 0.000971696 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.992914 | 0.993335 | ADAGRAD  | 6.10361e-05 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.992914 | 0.992362 | SGD      | 0.00659144  | **            | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.990142 | 0.993335 | ADAGRAD  | 0.000596607 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.990142 | 0.992362 | NOVOGRAD | 0.00141893  | **            | final_validation_auc |
| ADAGRAD       | NOVOGRAD      | 0.993335 | 0.992362 | ADAGRAD  | 0.00081131  | ***           | final_validation_auc |

