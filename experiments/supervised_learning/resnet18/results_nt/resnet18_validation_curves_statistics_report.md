# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 2.0591 | 0.0289 | 0.0129 | 2.0232 | 2.0951 |
| MILO_LW | 2.2688 | 0.0396 | 0.0177 | 2.2196 | 2.3180 |
| SGD | 2.1728 | 0.0633 | 0.0283 | 2.0943 | 2.2514 |
| ADAMW | 2.9832 | 0.2092 | 0.0936 | 2.7234 | 3.2430 |
| ADAGRAD | 2.1097 | 0.0269 | 0.0120 | 2.0763 | 2.1431 |
| NOVOGRAD | 2.6137 | 0.1040 | 0.0465 | 2.4846 | 2.7429 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  2.05915 |  2.2688  | MILO     | 2.15342e-05 | ***           | final_validation_loss |
| MILO          | SGD           |  2.05915 |  2.17283 | MILO     | 0.012015    | *             | final_validation_loss |
| MILO          | ADAMW         |  2.05915 |  2.98324 | MILO     | 0.000507494 | ***           | final_validation_loss |
| MILO          | ADAGRAD       |  2.05915 |  2.10972 | MILO     | 0.0212144   | *             | final_validation_loss |
| MILO          | NOVOGRAD      |  2.05915 |  2.61373 | MILO     | 0.000143588 | ***           | final_validation_loss |
| MILO_LW       | SGD           |  2.2688  |  2.17283 | SGD      | 0.0248967   | *             | final_validation_loss |
| MILO_LW       | ADAMW         |  2.2688  |  2.98324 | MILO_LW  | 0.00127816  | **            | final_validation_loss |
| MILO_LW       | ADAGRAD       |  2.2688  |  2.10972 | ADAGRAD  | 0.000141581 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      |  2.2688  |  2.61373 | MILO_LW  | 0.000861041 | ***           | final_validation_loss |
| SGD           | ADAMW         |  2.17283 |  2.98324 | SGD      | 0.000545413 | ***           | final_validation_loss |
| SGD           | ADAGRAD       |  2.17283 |  2.10972 | ADAGRAD  | 0.0911441   |               | final_validation_loss |
| SGD           | NOVOGRAD      |  2.17283 |  2.61373 | SGD      | 0.000115492 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       |  2.98324 |  2.10972 | ADAGRAD  | 0.00064773  | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      |  2.98324 |  2.61373 | NOVOGRAD | 0.0127557   | *             | final_validation_loss |
| ADAGRAD       | NOVOGRAD      |  2.10972 |  2.61373 | ADAGRAD  | 0.000238177 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 48.5541 | 0.4561 | 0.2040 | 47.9878 | 49.1204 |
| MILO_LW | 41.7333 | 0.7425 | 0.3321 | 40.8114 | 42.6553 |
| SGD | 49.0667 | 0.7870 | 0.3520 | 48.0895 | 50.0439 |
| ADAMW | 43.3807 | 2.1541 | 0.9633 | 40.7061 | 46.0553 |
| ADAGRAD | 47.4015 | 0.7340 | 0.3283 | 46.4901 | 48.3129 |
| NOVOGRAD | 43.1467 | 1.4528 | 0.6497 | 41.3428 | 44.9506 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  48.5541 |  41.7333 | MILO     | 8.33984e-07 | ***           | final_validation_accuracy |
| MILO          | SGD           |  48.5541 |  49.0667 | SGD      | 0.251539    |               | final_validation_accuracy |
| MILO          | ADAMW         |  48.5541 |  43.3807 | MILO     | 0.00493794  | **            | final_validation_accuracy |
| MILO          | ADAGRAD       |  48.5541 |  47.4015 | MILO     | 0.0215721   | *             | final_validation_accuracy |
| MILO          | NOVOGRAD      |  48.5541 |  43.1467 | MILO     | 0.000626779 | ***           | final_validation_accuracy |
| MILO_LW       | SGD           |  41.7333 |  49.0667 | SGD      | 3.68144e-07 | ***           | final_validation_accuracy |
| MILO_LW       | ADAMW         |  41.7333 |  43.3807 | ADAMW    | 0.167589    |               | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  41.7333 |  47.4015 | ADAGRAD  | 1.96555e-06 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  41.7333 |  43.1467 | NOVOGRAD | 0.101219    |               | final_validation_accuracy |
| SGD           | ADAMW         |  49.0667 |  43.3807 | SGD      | 0.00254189  | **            | final_validation_accuracy |
| SGD           | ADAGRAD       |  49.0667 |  47.4015 | SGD      | 0.00863282  | **            | final_validation_accuracy |
| SGD           | NOVOGRAD      |  49.0667 |  43.1467 | SGD      | 0.000176355 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  43.3807 |  47.4015 | ADAGRAD  | 0.011211    | *             | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  43.3807 |  43.1467 | ADAMW    | 0.846064    |               | final_validation_accuracy |
| ADAGRAD       | NOVOGRAD      |  47.4015 |  43.1467 | ADAGRAD  | 0.00116168  | **            | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.4827 | 0.0040 | 0.0018 | 0.4776 | 0.4877 |
| MILO_LW | 0.4088 | 0.0080 | 0.0036 | 0.3988 | 0.4187 |
| SGD | 0.4887 | 0.0075 | 0.0034 | 0.4793 | 0.4980 |
| ADAMW | 0.4328 | 0.0222 | 0.0099 | 0.4052 | 0.4604 |
| ADAGRAD | 0.4773 | 0.0073 | 0.0033 | 0.4682 | 0.4864 |
| NOVOGRAD | 0.4291 | 0.0144 | 0.0064 | 0.4112 | 0.4470 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.482667 | 0.408752 | MILO     | 1.84646e-06 | ***           | final_validation_f1_score |
| MILO          | SGD           | 0.482667 | 0.488671 | SGD      | 0.165857    |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.482667 | 0.43281  | MILO     | 0.00660849  | **            | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.482667 | 0.477276 | MILO     | 0.198653    |               | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.482667 | 0.429121 | MILO     | 0.000700143 | ***           | final_validation_f1_score |
| MILO_LW       | SGD           | 0.408752 | 0.488671 | SGD      | 2.11723e-07 | ***           | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.408752 | 0.43281  | ADAMW    | 0.0713077   |               | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.408752 | 0.477276 | ADAGRAD  | 6.59075e-07 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.408752 | 0.429121 | NOVOGRAD | 0.03125     | *             | final_validation_f1_score |
| SGD           | ADAMW         | 0.488671 | 0.43281  | SGD      | 0.00329766  | **            | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.488671 | 0.477276 | SGD      | 0.0415429   | *             | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.488671 | 0.429121 | SGD      | 0.000173012 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.43281  | 0.477276 | ADAGRAD  | 0.00857763  | **            | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.43281  | 0.429121 | ADAMW    | 0.764453    |               | final_validation_f1_score |
| ADAGRAD       | NOVOGRAD      | 0.477276 | 0.429121 | ADAGRAD  | 0.000573554 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9581 | 0.0012 | 0.0005 | 0.9566 | 0.9596 |
| MILO_LW | 0.9563 | 0.0016 | 0.0007 | 0.9543 | 0.9583 |
| SGD | 0.9646 | 0.0015 | 0.0007 | 0.9627 | 0.9664 |
| ADAMW | 0.9535 | 0.0044 | 0.0020 | 0.9481 | 0.9589 |
| ADAGRAD | 0.9630 | 0.0006 | 0.0003 | 0.9623 | 0.9638 |
| NOVOGRAD | 0.9553 | 0.0029 | 0.0013 | 0.9517 | 0.9589 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.958135 | 0.956303 | MILO     | 0.0782825   |               | final_validation_auc |
| MILO          | SGD           | 0.958135 | 0.964564 | SGD      | 9.05589e-05 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.958135 | 0.953516 | MILO     | 0.0763052   |               | final_validation_auc |
| MILO          | ADAGRAD       | 0.958135 | 0.963039 | ADAGRAD  | 0.000202063 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.958135 | 0.955325 | MILO     | 0.096681    |               | final_validation_auc |
| MILO_LW       | SGD           | 0.956303 | 0.964564 | SGD      | 3.09724e-05 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.956303 | 0.953516 | MILO_LW  | 0.237924    |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.956303 | 0.963039 | ADAGRAD  | 0.000265556 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.956303 | 0.955325 | MILO_LW  | 0.531088    |               | final_validation_auc |
| SGD           | ADAMW         | 0.964564 | 0.953516 | SGD      | 0.00321288  | **            | final_validation_auc |
| SGD           | ADAGRAD       | 0.964564 | 0.963039 | SGD      | 0.0862692   |               | final_validation_auc |
| SGD           | NOVOGRAD      | 0.964564 | 0.955325 | SGD      | 0.000699574 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.953516 | 0.963039 | ADAGRAD  | 0.00769876  | **            | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.953516 | 0.955325 | NOVOGRAD | 0.465333    |               | final_validation_auc |
| ADAGRAD       | NOVOGRAD      | 0.963039 | 0.955325 | ADAGRAD  | 0.00320546  | **            | final_validation_auc |

