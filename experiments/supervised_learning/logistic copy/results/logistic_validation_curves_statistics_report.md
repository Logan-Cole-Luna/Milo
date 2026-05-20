# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.3625 | 0.0042 | 0.0019 | 0.3573 | 0.3677 |
| MILO_LW | 0.3446 | 0.0014 | 0.0006 | 0.3428 | 0.3463 |
| SGD | 0.3076 | 0.0020 | 0.0009 | 0.3051 | 0.3100 |
| ADAMW | 0.6191 | 0.0407 | 0.0182 | 0.5686 | 0.6697 |
| ADAM_MINI | 0.6992 | 0.0328 | 0.0147 | 0.6585 | 0.7399 |
| NOVOGRAD | 0.3250 | 0.0093 | 0.0042 | 0.3134 | 0.3365 |
| ADAGRAD | 0.3987 | 0.0023 | 0.0010 | 0.3958 | 0.4016 |
| ADEMAMIX | 0.6075 | 0.0726 | 0.0325 | 0.5173 | 0.6976 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.362502 | 0.344573 | MILO_LW  | 0.000293988 | ***           | final_validation_loss |
| MILO          | SGD           | 0.362502 | 0.307551 | SGD      | 3.28394e-07 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.362502 | 0.619146 | MILO     | 0.000131694 | ***           | final_validation_loss |
| MILO          | ADAM_MINI     | 0.362502 | 0.699198 | MILO     | 1.68867e-05 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      | 0.362502 | 0.324979 | NOVOGRAD | 0.000257028 | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.362502 | 0.398713 | MILO     | 1.87662e-06 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      | 0.362502 | 0.607485 | MILO     | 0.00162091  | **            | final_validation_loss |
| MILO_LW       | SGD           | 0.344573 | 0.307551 | SGD      | 2.89493e-09 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         | 0.344573 | 0.619146 | MILO_LW  | 0.000111301 | ***           | final_validation_loss |
| MILO_LW       | ADAM_MINI     | 0.344573 | 0.699198 | MILO_LW  | 1.6859e-05  | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.344573 | 0.324979 | NOVOGRAD | 0.00857608  | **            | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.344573 | 0.398713 | MILO_LW  | 1.65329e-09 | ***           | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.344573 | 0.607485 | MILO_LW  | 0.00126176  | **            | final_validation_loss |
| SGD           | ADAMW         | 0.307551 | 0.619146 | SGD      | 6.653e-05   | ***           | final_validation_loss |
| SGD           | ADAM_MINI     | 0.307551 | 0.699198 | SGD      | 1.10303e-05 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 0.307551 | 0.324979 | SGD      | 0.0124079   | *             | final_validation_loss |
| SGD           | ADAGRAD       | 0.307551 | 0.398713 | SGD      | 4.46453e-12 | ***           | final_validation_loss |
| SGD           | ADEMAMIX      | 0.307551 | 0.607485 | SGD      | 0.000759477 | ***           | final_validation_loss |
| ADAMW         | ADAM_MINI     | 0.619146 | 0.699198 | ADAMW    | 0.00967136  | **            | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.619146 | 0.324979 | NOVOGRAD | 4.77201e-05 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.619146 | 0.398713 | ADAGRAD  | 0.000259336 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.619146 | 0.607485 | ADEMAMIX | 0.764259    |               | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      | 0.699198 | 0.324979 | NOVOGRAD | 4.32986e-06 | ***           | final_validation_loss |
| ADAM_MINI     | ADAGRAD       | 0.699198 | 0.398713 | ADAGRAD  | 3.13164e-05 | ***           | final_validation_loss |
| ADAM_MINI     | ADEMAMIX      | 0.699198 | 0.607485 | ADEMAMIX | 0.0450725   | *             | final_validation_loss |
| NOVOGRAD      | ADAGRAD       | 0.324979 | 0.398713 | NOVOGRAD | 2.8344e-05  | ***           | final_validation_loss |
| NOVOGRAD      | ADEMAMIX      | 0.324979 | 0.607485 | NOVOGRAD | 0.000857535 | ***           | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 0.398713 | 0.607485 | ADAGRAD  | 0.00299481  | **            | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 91.1506 | 0.1786 | 0.0799 | 90.9289 | 91.3723 |
| MILO_LW | 91.2543 | 0.1230 | 0.0550 | 91.1016 | 91.4071 |
| SGD | 91.4272 | 0.2247 | 0.1005 | 91.1481 | 91.7062 |
| ADAMW | 88.7877 | 0.8632 | 0.3860 | 87.7158 | 89.8595 |
| ADAM_MINI | 88.9111 | 0.4509 | 0.2016 | 88.3513 | 89.4709 |
| NOVOGRAD | 90.7877 | 0.2343 | 0.1048 | 90.4967 | 91.0786 |
| ADAGRAD | 89.3012 | 0.1110 | 0.0496 | 89.1634 | 89.4390 |
| ADEMAMIX | 88.6815 | 1.2374 | 0.5534 | 87.1451 | 90.2179 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better    |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:----------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  91.1506 |  91.2543 | MILO_LW   | 0.319901    |               | final_validation_accuracy |
| MILO          | SGD           |  91.1506 |  91.4272 | SGD       | 0.0650568   |               | final_validation_accuracy |
| MILO          | ADAMW         |  91.1506 |  88.7877 | MILO      | 0.00298613  | **            | final_validation_accuracy |
| MILO          | ADAM_MINI     |  91.1506 |  88.9111 | MILO      | 0.000113155 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  91.1506 |  90.7877 | MILO      | 0.0265388   | *             | final_validation_accuracy |
| MILO          | ADAGRAD       |  91.1506 |  89.3012 | MILO      | 3.60929e-07 | ***           | final_validation_accuracy |
| MILO          | ADEMAMIX      |  91.1506 |  88.6815 | MILO      | 0.010529    | *             | final_validation_accuracy |
| MILO_LW       | SGD           |  91.2543 |  91.4272 | SGD       | 0.180559    |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  91.2543 |  88.7877 | MILO_LW   | 0.00279088  | **            | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  91.2543 |  88.9111 | MILO_LW   | 0.000164752 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  91.2543 |  90.7877 | MILO_LW   | 0.00747357  | **            | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  91.2543 |  89.3012 | MILO_LW   | 5.36161e-09 | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  91.2543 |  88.6815 | MILO_LW   | 0.00938289  | **            | final_validation_accuracy |
| SGD           | ADAMW         |  91.4272 |  88.7877 | SGD       | 0.00170998  | **            | final_validation_accuracy |
| SGD           | ADAM_MINI     |  91.4272 |  88.9111 | SGD       | 3.55986e-05 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  91.4272 |  90.7877 | SGD       | 0.00228199  | **            | final_validation_accuracy |
| SGD           | ADAGRAD       |  91.4272 |  89.3012 | SGD       | 1.80134e-06 | ***           | final_validation_accuracy |
| SGD           | ADEMAMIX      |  91.4272 |  88.6815 | SGD       | 0.00691882  | **            | final_validation_accuracy |
| ADAMW         | ADAM_MINI     |  88.7877 |  88.9111 | ADAM_MINI | 0.786296    |               | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  88.7877 |  90.7877 | NOVOGRAD  | 0.00520215  | **            | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  88.7877 |  89.3012 | ADAGRAD   | 0.255352    |               | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  88.7877 |  88.6815 | ADAMW     | 0.879314    |               | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  88.9111 |  90.7877 | NOVOGRAD  | 0.000168498 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  88.9111 |  89.3012 | ADAGRAD   | 0.125743    |               | final_validation_accuracy |
| ADAM_MINI     | ADEMAMIX      |  88.9111 |  88.6815 | ADAM_MINI | 0.712535    |               | final_validation_accuracy |
| NOVOGRAD      | ADAGRAD       |  90.7877 |  89.3012 | NOVOGRAD  | 2.00797e-05 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADEMAMIX      |  90.7877 |  88.6815 | NOVOGRAD  | 0.0177714   | *             | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  89.3012 |  88.6815 | ADAGRAD   | 0.32621     |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9108 | 0.0018 | 0.0008 | 0.9085 | 0.9131 |
| MILO_LW | 0.9120 | 0.0012 | 0.0005 | 0.9104 | 0.9135 |
| SGD | 0.9135 | 0.0022 | 0.0010 | 0.9108 | 0.9163 |
| ADAMW | 0.8872 | 0.0084 | 0.0037 | 0.8768 | 0.8976 |
| ADAM_MINI | 0.8879 | 0.0043 | 0.0019 | 0.8825 | 0.8932 |
| NOVOGRAD | 0.9072 | 0.0022 | 0.0010 | 0.9045 | 0.9100 |
| ADAGRAD | 0.8920 | 0.0011 | 0.0005 | 0.8906 | 0.8934 |
| ADEMAMIX | 0.8863 | 0.0118 | 0.0053 | 0.8716 | 0.9010 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better    |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:----------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.910775 | 0.911957 | MILO_LW   | 0.269873    |               | final_validation_f1_score |
| MILO          | SGD           | 0.910775 | 0.913525 | SGD       | 0.0664368   |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.910775 | 0.88723  | MILO      | 0.00263241  | **            | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.910775 | 0.887855 | MILO      | 7.17228e-05 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.910775 | 0.907233 | MILO      | 0.0250208   | *             | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.910775 | 0.891966 | MILO      | 4.11763e-07 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.910775 | 0.886321 | MILO      | 0.00916405  | **            | final_validation_f1_score |
| MILO_LW       | SGD           | 0.911957 | 0.913525 | SGD       | 0.213677    |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.911957 | 0.88723  | MILO_LW   | 0.00245472  | **            | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.911957 | 0.887855 | MILO_LW   | 0.000117475 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.911957 | 0.907233 | MILO_LW   | 0.00511345  | **            | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.911957 | 0.891966 | MILO_LW   | 4.27993e-09 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.911957 | 0.886321 | MILO_LW   | 0.00805191  | **            | final_validation_f1_score |
| SGD           | ADAMW         | 0.913525 | 0.88723  | SGD       | 0.0015164   | **            | final_validation_f1_score |
| SGD           | ADAM_MINI     | 0.913525 | 0.887855 | SGD       | 2.36879e-05 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.913525 | 0.907233 | SGD       | 0.00197494  | **            | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.913525 | 0.891966 | SGD       | 1.3732e-06  | ***           | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.913525 | 0.886321 | SGD       | 0.00598655  | **            | final_validation_f1_score |
| ADAMW         | ADAM_MINI     | 0.88723  | 0.887855 | ADAM_MINI | 0.887095    |               | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.88723  | 0.907233 | NOVOGRAD  | 0.00468774  | **            | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.88723  | 0.891966 | ADAGRAD   | 0.276339    |               | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.88723  | 0.886321 | ADAMW     | 0.892296    |               | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.887855 | 0.907233 | NOVOGRAD  | 0.000119822 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.887855 | 0.891966 | ADAGRAD   | 0.101226    |               | final_validation_f1_score |
| ADAM_MINI     | ADEMAMIX      | 0.887855 | 0.886321 | ADAM_MINI | 0.796067    |               | final_validation_f1_score |
| NOVOGRAD      | ADAGRAD       | 0.907233 | 0.891966 | NOVOGRAD  | 9.13539e-06 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADEMAMIX      | 0.907233 | 0.886321 | NOVOGRAD  | 0.0155781   | *             | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.891966 | 0.886321 | ADAGRAD   | 0.346586    |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9918 | 0.0001 | 0.0001 | 0.9916 | 0.9920 |
| MILO_LW | 0.9921 | 0.0001 | 0.0000 | 0.9920 | 0.9922 |
| SGD | 0.9926 | 0.0001 | 0.0001 | 0.9925 | 0.9928 |
| ADAMW | 0.9901 | 0.0007 | 0.0003 | 0.9892 | 0.9910 |
| ADAM_MINI | 0.9900 | 0.0002 | 0.0001 | 0.9898 | 0.9902 |
| NOVOGRAD | 0.9921 | 0.0003 | 0.0001 | 0.9918 | 0.9925 |
| ADAGRAD | 0.9901 | 0.0002 | 0.0001 | 0.9899 | 0.9903 |
| ADEMAMIX | 0.9902 | 0.0007 | 0.0003 | 0.9893 | 0.9911 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.991813 | 0.992096 | MILO_LW  | 0.0104439   | *             | final_validation_auc |
| MILO          | SGD           | 0.991813 | 0.992646 | SGD      | 1.43559e-05 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.991813 | 0.990088 | MILO     | 0.00457398  | **            | final_validation_auc |
| MILO          | ADAM_MINI     | 0.991813 | 0.990016 | MILO     | 1.79604e-07 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.991813 | 0.992124 | NOVOGRAD | 0.0651529   |               | final_validation_auc |
| MILO          | ADAGRAD       | 0.991813 | 0.99007  | MILO     | 8.22095e-08 | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.991813 | 0.990178 | MILO     | 0.00585367  | **            | final_validation_auc |
| MILO_LW       | SGD           | 0.992096 | 0.992646 | SGD      | 0.000115438 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.992096 | 0.990088 | MILO_LW  | 0.00301817  | **            | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.992096 | 0.990016 | MILO_LW  | 3.29913e-06 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.992096 | 0.992124 | NOVOGRAD | 0.827667    |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.992096 | 0.99007  | MILO_LW  | 1.05058e-06 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.992096 | 0.990178 | MILO_LW  | 0.00371555  | **            | final_validation_auc |
| SGD           | ADAMW         | 0.992646 | 0.990088 | SGD      | 0.00104409  | **            | final_validation_auc |
| SGD           | ADAM_MINI     | 0.992646 | 0.990016 | SGD      | 2.41434e-08 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.992646 | 0.992124 | SGD      | 0.00940062  | **            | final_validation_auc |
| SGD           | ADAGRAD       | 0.992646 | 0.99007  | SGD      | 4.05714e-09 | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.992646 | 0.990178 | SGD      | 0.00125658  | **            | final_validation_auc |
| ADAMW         | ADAM_MINI     | 0.990088 | 0.990016 | ADAMW    | 0.835201    |               | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.990088 | 0.992124 | NOVOGRAD | 0.00160317  | **            | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.990088 | 0.99007  | ADAMW    | 0.959244    |               | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.990088 | 0.990178 | ADEMAMIX | 0.844586    |               | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.990016 | 0.992124 | NOVOGRAD | 2.08065e-06 | ***           | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.990016 | 0.99007  | ADAGRAD  | 0.619265    |               | final_validation_auc |
| ADAM_MINI     | ADEMAMIX      | 0.990016 | 0.990178 | ADEMAMIX | 0.643889    |               | final_validation_auc |
| NOVOGRAD      | ADAGRAD       | 0.992124 | 0.99007  | NOVOGRAD | 4.33956e-06 | ***           | final_validation_auc |
| NOVOGRAD      | ADEMAMIX      | 0.992124 | 0.990178 | NOVOGRAD | 0.00208027  | **            | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.99007  | 0.990178 | ADEMAMIX | 0.754989    |               | final_validation_auc |

