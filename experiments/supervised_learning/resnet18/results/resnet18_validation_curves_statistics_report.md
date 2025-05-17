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
| SGD | 2.4128 | 0.1821 | 0.0815 | 2.1866 | 2.6389 |
| ADAMW | 3.3230 | 0.1912 | 0.0855 | 3.0855 | 3.5604 |
| ADAGRAD | 2.1847 | 0.0610 | 0.0273 | 2.1091 | 2.2604 |
| NOVOGRAD | 2.3787 | 0.1244 | 0.0556 | 2.2242 | 2.5331 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  2.05915 |  2.2688  | MILO     | 2.15342e-05 | ***           | final_validation_loss |
| MILO          | SGD           |  2.05915 |  2.41276 | MILO     | 0.0114752   | *             | final_validation_loss |
| MILO          | ADAMW         |  2.05915 |  3.32297 | MILO     | 9.51609e-05 | ***           | final_validation_loss |
| MILO          | ADAGRAD       |  2.05915 |  2.18474 | MILO     | 0.00659461  | **            | final_validation_loss |
| MILO          | NOVOGRAD      |  2.05915 |  2.37865 | MILO     | 0.00367637  | **            | final_validation_loss |
| MILO_LW       | SGD           |  2.2688  |  2.41276 | MILO_LW  | 0.153027    |               | final_validation_loss |
| MILO_LW       | ADAMW         |  2.2688  |  3.32297 | MILO_LW  | 0.000166739 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       |  2.2688  |  2.18474 | ADAGRAD  | 0.0367799   | *             | final_validation_loss |
| MILO_LW       | NOVOGRAD      |  2.2688  |  2.37865 | MILO_LW  | 0.121011    |               | final_validation_loss |
| SGD           | ADAMW         |  2.41276 |  3.32297 | SGD      | 5.7788e-05  | ***           | final_validation_loss |
| SGD           | ADAGRAD       |  2.41276 |  2.18474 | ADAGRAD  | 0.0462325   | *             | final_validation_loss |
| SGD           | NOVOGRAD      |  2.41276 |  2.37865 | NOVOGRAD | 0.739568    |               | final_validation_loss |
| ADAMW         | ADAGRAD       |  3.32297 |  2.18474 | ADAGRAD  | 7.07456e-05 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      |  3.32297 |  2.37865 | NOVOGRAD | 3.9865e-05  | ***           | final_validation_loss |
| ADAGRAD       | NOVOGRAD      |  2.18474 |  2.37865 | ADAGRAD  | 0.0211755   | *             | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 48.5541 | 0.4561 | 0.2040 | 47.9878 | 49.1204 |
| MILO_LW | 41.7333 | 0.7425 | 0.3321 | 40.8114 | 42.6553 |
| SGD | 43.1289 | 2.0720 | 0.9266 | 40.5562 | 45.7016 |
| ADAMW | 41.8726 | 1.4728 | 0.6587 | 40.0439 | 43.7013 |
| ADAGRAD | 44.4711 | 1.2384 | 0.5538 | 42.9335 | 46.0087 |
| NOVOGRAD | 48.9304 | 1.3081 | 0.5850 | 47.3061 | 50.5546 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  48.5541 |  41.7333 | MILO     | 8.33984e-07 | ***           | final_validation_accuracy |
| MILO          | SGD           |  48.5541 |  43.1289 | MILO     | 0.00348006  | **            | final_validation_accuracy |
| MILO          | ADAMW         |  48.5541 |  41.8726 | MILO     | 0.000259774 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  48.5541 |  44.4711 | MILO     | 0.000917937 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  48.5541 |  48.9304 | NOVOGRAD | 0.570344    |               | final_validation_accuracy |
| MILO_LW       | SGD           |  41.7333 |  43.1289 | SGD      | 0.215334    |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  41.7333 |  41.8726 | ADAMW    | 0.856572    |               | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  41.7333 |  44.4711 | ADAGRAD  | 0.0044694   | **            | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  41.7333 |  48.9304 | NOVOGRAD | 2.73752e-05 | ***           | final_validation_accuracy |
| SGD           | ADAMW         |  43.1289 |  41.8726 | SGD      | 0.304596    |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  43.1289 |  44.4711 | ADAGRAD  | 0.256487    |               | final_validation_accuracy |
| SGD           | NOVOGRAD      |  43.1289 |  48.9304 | NOVOGRAD | 0.00126899  | **            | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  41.8726 |  44.4711 | ADAGRAD  | 0.0171391   | *             | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  41.8726 |  48.9304 | NOVOGRAD | 4.6727e-05  | ***           | final_validation_accuracy |
| ADAGRAD       | NOVOGRAD      |  44.4711 |  48.9304 | NOVOGRAD | 0.00055605  | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.4827 | 0.0040 | 0.0018 | 0.4776 | 0.4877 |
| MILO_LW | 0.4088 | 0.0080 | 0.0036 | 0.3988 | 0.4187 |
| SGD | 0.4286 | 0.0190 | 0.0085 | 0.4050 | 0.4521 |
| ADAMW | 0.4163 | 0.0156 | 0.0070 | 0.3969 | 0.4357 |
| ADAGRAD | 0.4392 | 0.0133 | 0.0059 | 0.4228 | 0.4557 |
| NOVOGRAD | 0.4885 | 0.0127 | 0.0057 | 0.4727 | 0.5043 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.482667 | 0.408752 | MILO     | 1.84646e-06 | ***           | final_validation_f1_score |
| MILO          | SGD           | 0.482667 | 0.428582 | MILO     | 0.00251247  | **            | final_validation_f1_score |
| MILO          | ADAMW         | 0.482667 | 0.416311 | MILO     | 0.000426231 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.482667 | 0.439214 | MILO     | 0.00112967  | **            | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.482667 | 0.488506 | NOVOGRAD | 0.373998    |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.408752 | 0.428582 | SGD      | 0.0799827   |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.408752 | 0.416311 | ADAMW    | 0.373502    |               | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.408752 | 0.439214 | ADAGRAD  | 0.00367142  | **            | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.408752 | 0.488506 | NOVOGRAD | 9.09224e-06 | ***           | final_validation_f1_score |
| SGD           | ADAMW         | 0.428582 | 0.416311 | SGD      | 0.298061    |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.428582 | 0.439214 | ADAGRAD  | 0.337739    |               | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.428582 | 0.488506 | NOVOGRAD | 0.000623145 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.416311 | 0.439214 | ADAGRAD  | 0.0378567   | *             | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.416311 | 0.488506 | NOVOGRAD | 5.46317e-05 | ***           | final_validation_f1_score |
| ADAGRAD       | NOVOGRAD      | 0.439214 | 0.488506 | NOVOGRAD | 0.000323955 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9581 | 0.0012 | 0.0005 | 0.9566 | 0.9596 |
| MILO_LW | 0.9563 | 0.0016 | 0.0007 | 0.9543 | 0.9583 |
| SGD | 0.9569 | 0.0048 | 0.0022 | 0.9509 | 0.9629 |
| ADAMW | 0.9467 | 0.0055 | 0.0025 | 0.9399 | 0.9536 |
| ADAGRAD | 0.9562 | 0.0028 | 0.0013 | 0.9527 | 0.9597 |
| NOVOGRAD | 0.9656 | 0.0027 | 0.0012 | 0.9623 | 0.9689 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.958135 | 0.956303 | MILO     | 0.0782825   |               | final_validation_auc |
| MILO          | SGD           | 0.958135 | 0.956912 | MILO     | 0.610127    |               | final_validation_auc |
| MILO          | ADAMW         | 0.958135 | 0.946728 | MILO     | 0.00860623  | **            | final_validation_auc |
| MILO          | ADAGRAD       | 0.958135 | 0.956164 | MILO     | 0.206515    |               | final_validation_auc |
| MILO          | NOVOGRAD      | 0.958135 | 0.96561  | NOVOGRAD | 0.00157778  | **            | final_validation_auc |
| MILO_LW       | SGD           | 0.956303 | 0.956912 | SGD      | 0.800455    |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.956303 | 0.946728 | MILO_LW  | 0.0153812   | *             | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.956303 | 0.956164 | MILO_LW  | 0.926901    |               | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.956303 | 0.96561  | NOVOGRAD | 0.000366833 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.956912 | 0.946728 | SGD      | 0.0149267   | *             | final_validation_auc |
| SGD           | ADAGRAD       | 0.956912 | 0.956164 | SGD      | 0.774971    |               | final_validation_auc |
| SGD           | NOVOGRAD      | 0.956912 | 0.96561  | NOVOGRAD | 0.0118716   | *             | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.946728 | 0.956164 | ADAGRAD  | 0.0145308   | *             | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.946728 | 0.96561  | NOVOGRAD | 0.000543793 | ***           | final_validation_auc |
| ADAGRAD       | NOVOGRAD      | 0.956164 | 0.96561  | NOVOGRAD | 0.000620674 | ***           | final_validation_auc |

