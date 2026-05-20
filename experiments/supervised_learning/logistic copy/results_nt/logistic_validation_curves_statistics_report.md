# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.3509 | 0.0041 | 0.0018 | 0.3458 | 0.3560 |
| MILO_LW | 0.3507 | 0.0016 | 0.0007 | 0.3487 | 0.3527 |
| SGD | 0.2973 | 0.0018 | 0.0008 | 0.2952 | 0.2995 |
| ADAMW | 0.5517 | 0.0480 | 0.0215 | 0.4921 | 0.6113 |
| ADAM_MINI | 0.7336 | 0.0442 | 0.0198 | 0.6787 | 0.7884 |
| NOVOGRAD | 0.3201 | 0.0111 | 0.0049 | 0.3064 | 0.3339 |
| ADAGRAD | 0.2855 | 0.0010 | 0.0005 | 0.2843 | 0.2868 |
| ADEMAMIX | 0.4934 | 0.0572 | 0.0256 | 0.4224 | 0.5644 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.350913 | 0.350687 | MILO_LW  | 0.913139    |               | final_validation_loss |
| MILO          | SGD           | 0.350913 | 0.297345 | SGD      | 5.51332e-07 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.350913 | 0.551702 | MILO     | 0.000686893 | ***           | final_validation_loss |
| MILO          | ADAM_MINI     | 0.350913 | 0.733556 | MILO     | 3.75663e-05 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      | 0.350913 | 0.320129 | NOVOGRAD | 0.00198385  | **            | final_validation_loss |
| MILO          | ADAGRAD       | 0.350913 | 0.28555  | ADAGRAD  | 1.24301e-06 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      | 0.350913 | 0.493389 | MILO     | 0.00497992  | **            | final_validation_loss |
| MILO_LW       | SGD           | 0.350687 | 0.297345 | SGD      | 3.26947e-11 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         | 0.350687 | 0.551702 | MILO_LW  | 0.000716338 | ***           | final_validation_loss |
| MILO_LW       | ADAM_MINI     | 0.350687 | 0.733556 | MILO_LW  | 4.1182e-05  | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.350687 | 0.320129 | NOVOGRAD | 0.00315473  | **            | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.350687 | 0.28555  | ADAGRAD  | 3.94342e-11 | ***           | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.350687 | 0.493389 | MILO_LW  | 0.00503922  | **            | final_validation_loss |
| SGD           | ADAMW         | 0.297345 | 0.551702 | SGD      | 0.000286115 | ***           | final_validation_loss |
| SGD           | ADAM_MINI     | 0.297345 | 0.733556 | SGD      | 2.44258e-05 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 0.297345 | 0.320129 | SGD      | 0.00927835  | **            | final_validation_loss |
| SGD           | ADAGRAD       | 0.297345 | 0.28555  | ADAGRAD  | 7.59676e-06 | ***           | final_validation_loss |
| SGD           | ADEMAMIX      | 0.297345 | 0.493389 | SGD      | 0.00154714  | **            | final_validation_loss |
| ADAMW         | ADAM_MINI     | 0.551702 | 0.733556 | ADAMW    | 0.000257096 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.551702 | 0.320129 | NOVOGRAD | 0.000269345 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.551702 | 0.28555  | ADAGRAD  | 0.000241593 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.551702 | 0.493389 | ADEMAMIX | 0.119914    |               | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      | 0.733556 | 0.320129 | NOVOGRAD | 1.35189e-05 | ***           | final_validation_loss |
| ADAM_MINI     | ADAGRAD       | 0.733556 | 0.28555  | ADAGRAD  | 2.22944e-05 | ***           | final_validation_loss |
| ADAM_MINI     | ADEMAMIX      | 0.733556 | 0.493389 | ADEMAMIX | 0.000101479 | ***           | final_validation_loss |
| NOVOGRAD      | ADAGRAD       | 0.320129 | 0.28555  | ADAGRAD  | 0.00210015  | **            | final_validation_loss |
| NOVOGRAD      | ADEMAMIX      | 0.320129 | 0.493389 | NOVOGRAD | 0.00204589  | **            | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 0.28555  | 0.493389 | ADAGRAD  | 0.00124389  | **            | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 91.3531 | 0.1337 | 0.0598 | 91.1871 | 91.5191 |
| MILO_LW | 91.2370 | 0.1345 | 0.0602 | 91.0700 | 91.4040 |
| SGD | 91.7951 | 0.2395 | 0.1071 | 91.4977 | 92.0924 |
| ADAMW | 88.8667 | 0.9555 | 0.4273 | 87.6803 | 90.0530 |
| ADAM_MINI | 88.8568 | 0.7127 | 0.3187 | 87.9718 | 89.7418 |
| NOVOGRAD | 90.9481 | 0.3658 | 0.1636 | 90.4940 | 91.4023 |
| ADAGRAD | 92.2123 | 0.0548 | 0.0245 | 92.1443 | 92.2804 |
| ADEMAMIX | 88.9704 | 0.8612 | 0.3851 | 87.9010 | 90.0397 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  91.3531 |  91.237  | MILO     | 0.208434    |               | final_validation_accuracy |
| MILO          | SGD           |  91.3531 |  91.7951 | SGD      | 0.0104928   | *             | final_validation_accuracy |
| MILO          | ADAMW         |  91.3531 |  88.8667 | MILO     | 0.00399584  | **            | final_validation_accuracy |
| MILO          | ADAM_MINI     |  91.3531 |  88.8568 | MILO     | 0.00115795  | **            | final_validation_accuracy |
| MILO          | NOVOGRAD      |  91.3531 |  90.9481 | MILO     | 0.0671025   |               | final_validation_accuracy |
| MILO          | ADAGRAD       |  91.3531 |  92.2123 | ADAGRAD  | 2.81609e-05 | ***           | final_validation_accuracy |
| MILO          | ADEMAMIX      |  91.3531 |  88.9704 | MILO     | 0.00310486  | **            | final_validation_accuracy |
| MILO_LW       | SGD           |  91.237  |  91.7951 | SGD      | 0.00347674  | **            | final_validation_accuracy |
| MILO_LW       | ADAMW         |  91.237  |  88.8667 | MILO_LW  | 0.00477608  | **            | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  91.237  |  88.8568 | MILO_LW  | 0.00139937  | **            | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  91.237  |  90.9481 | MILO_LW  | 0.157572    |               | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  91.237  |  92.2123 | ADAGRAD  | 1.53132e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  91.237  |  88.9704 | MILO_LW  | 0.00375286  | **            | final_validation_accuracy |
| SGD           | ADAMW         |  91.7951 |  88.8667 | SGD      | 0.00173213  | **            | final_validation_accuracy |
| SGD           | ADAM_MINI     |  91.7951 |  88.8568 | SGD      | 0.000362882 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  91.7951 |  90.9481 | SGD      | 0.00355172  | **            | final_validation_accuracy |
| SGD           | ADAGRAD       |  91.7951 |  92.2123 | ADAGRAD  | 0.0159358   | *             | final_validation_accuracy |
| SGD           | ADEMAMIX      |  91.7951 |  88.9704 | SGD      | 0.00121582  | **            | final_validation_accuracy |
| ADAMW         | ADAM_MINI     |  88.8667 |  88.8568 | ADAMW    | 0.985708    |               | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  88.8667 |  90.9481 | NOVOGRAD | 0.00569596  | **            | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  88.8667 |  92.2123 | ADAGRAD  | 0.00140704  | **            | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  88.8667 |  88.9704 | ADEMAMIX | 0.861469    |               | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  88.8568 |  90.9481 | NOVOGRAD | 0.00113316  | **            | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  88.8568 |  92.2123 | ADAGRAD  | 0.000438034 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADEMAMIX      |  88.8568 |  88.9704 | ADEMAMIX | 0.826169    |               | final_validation_accuracy |
| NOVOGRAD      | ADAGRAD       |  90.9481 |  92.2123 | ADAGRAD  | 0.0013161   | **            | final_validation_accuracy |
| NOVOGRAD      | ADEMAMIX      |  90.9481 |  88.9704 | NOVOGRAD | 0.00427541  | **            | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  92.2123 |  88.9704 | ADAGRAD  | 0.00106081  | **            | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9128 | 0.0014 | 0.0006 | 0.9111 | 0.9146 |
| MILO_LW | 0.9118 | 0.0013 | 0.0006 | 0.9101 | 0.9134 |
| SGD | 0.9173 | 0.0024 | 0.0011 | 0.9143 | 0.9203 |
| ADAMW | 0.8882 | 0.0091 | 0.0041 | 0.8769 | 0.8994 |
| ADAM_MINI | 0.8875 | 0.0069 | 0.0031 | 0.8789 | 0.8961 |
| NOVOGRAD | 0.9088 | 0.0036 | 0.0016 | 0.9043 | 0.9132 |
| ADAGRAD | 0.9215 | 0.0005 | 0.0002 | 0.9208 | 0.9221 |
| ADEMAMIX | 0.8886 | 0.0084 | 0.0038 | 0.8782 | 0.8991 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.912836 | 0.911758 | MILO     | 0.247852    |               | final_validation_f1_score |
| MILO          | SGD           | 0.912836 | 0.917262 | SGD      | 0.0109154   | *             | final_validation_f1_score |
| MILO          | ADAMW         | 0.912836 | 0.888157 | MILO     | 0.00329839  | **            | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.912836 | 0.887462 | MILO     | 0.000920799 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.912836 | 0.908781 | MILO     | 0.063597    |               | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.912836 | 0.921468 | ADAGRAD  | 4.58253e-05 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.912836 | 0.888634 | MILO     | 0.00258402  | **            | final_validation_f1_score |
| MILO_LW       | SGD           | 0.911758 | 0.917262 | SGD      | 0.00394118  | **            | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.911758 | 0.888157 | MILO_LW  | 0.0039792   | **            | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.911758 | 0.887462 | MILO_LW  | 0.00114745  | **            | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.911758 | 0.908781 | MILO_LW  | 0.141485    |               | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.911758 | 0.921468 | ADAGRAD  | 1.21218e-05 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.911758 | 0.888634 | MILO_LW  | 0.00315123  | **            | final_validation_f1_score |
| SGD           | ADAMW         | 0.917262 | 0.888157 | SGD      | 0.00137507  | **            | final_validation_f1_score |
| SGD           | ADAM_MINI     | 0.917262 | 0.887462 | SGD      | 0.000281336 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.917262 | 0.908781 | SGD      | 0.00322479  | **            | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.917262 | 0.921468 | ADAGRAD  | 0.0160222   | *             | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.917262 | 0.888634 | SGD      | 0.000997369 | ***           | final_validation_f1_score |
| ADAMW         | ADAM_MINI     | 0.888157 | 0.887462 | ADAMW    | 0.895272    |               | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.888157 | 0.908781 | NOVOGRAD | 0.00464811  | **            | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.888157 | 0.921468 | ADAGRAD  | 0.0011693   | **            | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.888157 | 0.888634 | ADEMAMIX | 0.933323    |               | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.887462 | 0.908781 | NOVOGRAD | 0.000870422 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.887462 | 0.921468 | ADAGRAD  | 0.000369699 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADEMAMIX      | 0.887462 | 0.888634 | ADEMAMIX | 0.815889    |               | final_validation_f1_score |
| NOVOGRAD      | ADAGRAD       | 0.908781 | 0.921468 | ADAGRAD  | 0.00120321  | **            | final_validation_f1_score |
| NOVOGRAD      | ADEMAMIX      | 0.908781 | 0.888634 | NOVOGRAD | 0.00347689  | **            | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.921468 | 0.888634 | ADAGRAD  | 0.000912042 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9921 | 0.0001 | 0.0001 | 0.9920 | 0.9922 |
| MILO_LW | 0.9921 | 0.0001 | 0.0000 | 0.9920 | 0.9922 |
| SGD | 0.9929 | 0.0001 | 0.0000 | 0.9928 | 0.9930 |
| ADAMW | 0.9901 | 0.0007 | 0.0003 | 0.9892 | 0.9911 |
| ADAM_MINI | 0.9902 | 0.0005 | 0.0002 | 0.9896 | 0.9908 |
| NOVOGRAD | 0.9924 | 0.0003 | 0.0001 | 0.9920 | 0.9927 |
| ADAGRAD | 0.9933 | 0.0001 | 0.0000 | 0.9932 | 0.9934 |
| ADEMAMIX | 0.9904 | 0.0009 | 0.0004 | 0.9893 | 0.9915 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better    |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:----------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.992097 | 0.992085 | MILO      | 0.832092    |               | final_validation_auc |
| MILO          | SGD           | 0.992097 | 0.992914 | SGD       | 2.24259e-06 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.992097 | 0.990142 | MILO      | 0.00359154  | **            | final_validation_auc |
| MILO          | ADAM_MINI     | 0.992097 | 0.990222 | MILO      | 0.000680245 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.992097 | 0.992362 | NOVOGRAD  | 0.0860868   |               | final_validation_auc |
| MILO          | ADAGRAD       | 0.992097 | 0.993311 | ADAGRAD   | 1.95272e-07 | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.992097 | 0.990395 | MILO      | 0.0121301   | *             | final_validation_auc |
| MILO_LW       | SGD           | 0.992085 | 0.992914 | SGD       | 4.50194e-07 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.992085 | 0.990142 | MILO_LW   | 0.00394024  | **            | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.992085 | 0.990222 | MILO_LW   | 0.00088511  | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.992085 | 0.992362 | NOVOGRAD  | 0.0738045   |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.992085 | 0.993311 | ADAGRAD   | 5.22742e-09 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.992085 | 0.990395 | MILO_LW   | 0.0127962   | *             | final_validation_auc |
| SGD           | ADAMW         | 0.992914 | 0.990142 | SGD       | 0.000971782 | ***           | final_validation_auc |
| SGD           | ADAM_MINI     | 0.992914 | 0.990222 | SGD       | 0.000173412 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.992914 | 0.992362 | SGD       | 0.00659113  | **            | final_validation_auc |
| SGD           | ADAGRAD       | 0.992914 | 0.993311 | ADAGRAD   | 8.09932e-05 | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.992914 | 0.990395 | SGD       | 0.0029726   | **            | final_validation_auc |
| ADAMW         | ADAM_MINI     | 0.990142 | 0.990222 | ADAM_MINI | 0.844464    |               | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.990142 | 0.992362 | NOVOGRAD  | 0.0014191   | **            | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.990142 | 0.993311 | ADAGRAD   | 0.000591432 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.990142 | 0.990395 | ADEMAMIX  | 0.636212    |               | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.990222 | 0.992362 | NOVOGRAD  | 0.000113366 | ***           | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.990222 | 0.993311 | ADAGRAD   | 0.000107602 | ***           | final_validation_auc |
| ADAM_MINI     | ADEMAMIX      | 0.990222 | 0.990395 | ADEMAMIX  | 0.714182    |               | final_validation_auc |
| NOVOGRAD      | ADAGRAD       | 0.992362 | 0.993311 | ADAGRAD   | 0.000719839 | ***           | final_validation_auc |
| NOVOGRAD      | ADEMAMIX      | 0.992362 | 0.990395 | NOVOGRAD  | 0.00595255  | **            | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.993311 | 0.990395 | ADAGRAD   | 0.00173257  | **            | final_validation_auc |

