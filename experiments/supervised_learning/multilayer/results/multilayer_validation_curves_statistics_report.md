# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.2259 | 0.0182 | 0.0081 | 0.2034 | 0.2485 |
| MILO_LW | 0.1507 | 0.0102 | 0.0045 | 0.1381 | 0.1634 |
| SGD | 0.1047 | 0.0013 | 0.0006 | 0.1031 | 0.1062 |
| ADAMW | 0.1215 | 0.0097 | 0.0043 | 0.1095 | 0.1335 |
| ADAGRAD | 0.2156 | 0.0113 | 0.0051 | 0.2015 | 0.2297 |
| NOVOGRAD | 0.0872 | 0.0049 | 0.0022 | 0.0811 | 0.0933 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |    Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|----------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.225943 | 0.150747  | MILO_LW  | 0.000151986 | ***           | final_validation_loss |
| MILO          | SGD           | 0.225943 | 0.104657  | SGD      | 0.000110944 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.225943 | 0.121495  | ADAMW    | 2.48203e-05 | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.225943 | 0.215592  | ADAGRAD  | 0.316927    |               | final_validation_loss |
| MILO          | NOVOGRAD      | 0.225943 | 0.0872035 | NOVOGRAD | 2.96707e-05 | ***           | final_validation_loss |
| MILO_LW       | SGD           | 0.150747 | 0.104657  | SGD      | 0.00047008  | ***           | final_validation_loss |
| MILO_LW       | ADAMW         | 0.150747 | 0.121495  | ADAMW    | 0.00164005  | **            | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.150747 | 0.215592  | MILO_LW  | 1.31936e-05 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.150747 | 0.0872035 | NOVOGRAD | 2.06945e-05 | ***           | final_validation_loss |
| SGD           | ADAMW         | 0.104657 | 0.121495  | SGD      | 0.0171513   | *             | final_validation_loss |
| SGD           | ADAGRAD       | 0.104657 | 0.215592  | SGD      | 2.16145e-05 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 0.104657 | 0.0872035 | NOVOGRAD | 0.000902238 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.121495 | 0.215592  | ADAMW    | 7.75432e-07 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.121495 | 0.0872035 | NOVOGRAD | 0.00042865  | ***           | final_validation_loss |
| ADAGRAD       | NOVOGRAD      | 0.215592 | 0.0872035 | NOVOGRAD | 1.16408e-06 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 97.3210 | 0.1654 | 0.0740 | 97.1156 | 97.5264 |
| MILO_LW | 96.8198 | 0.1326 | 0.0593 | 96.6552 | 96.9844 |
| SGD | 97.1062 | 0.0862 | 0.0386 | 96.9991 | 97.2133 |
| ADAMW | 97.2914 | 0.2302 | 0.1030 | 97.0055 | 97.5772 |
| ADAGRAD | 94.1407 | 0.3818 | 0.1707 | 93.6667 | 94.6148 |
| NOVOGRAD | 97.3235 | 0.2001 | 0.0895 | 97.0750 | 97.5719 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  97.321  |  96.8198 | MILO     | 0.000859848 | ***           | final_validation_accuracy |
| MILO          | SGD           |  97.321  |  97.1062 | MILO     | 0.0418918   | *             | final_validation_accuracy |
| MILO          | ADAMW         |  97.321  |  97.2914 | MILO     | 0.821668    |               | final_validation_accuracy |
| MILO          | ADAGRAD       |  97.321  |  94.1407 | MILO     | 6.06257e-06 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  97.321  |  97.3235 | NOVOGRAD | 0.983572    |               | final_validation_accuracy |
| MILO_LW       | SGD           |  96.8198 |  97.1062 | SGD      | 0.00506563  | **            | final_validation_accuracy |
| MILO_LW       | ADAMW         |  96.8198 |  97.2914 | ADAMW    | 0.00649202  | **            | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  96.8198 |  94.1407 | MILO_LW  | 2.72214e-05 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  96.8198 |  97.3235 | NOVOGRAD | 0.00227636  | **            | final_validation_accuracy |
| SGD           | ADAMW         |  97.1062 |  97.2914 | ADAMW    | 0.151798    |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  97.1062 |  94.1407 | SGD      | 3.51853e-05 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  97.1062 |  97.3235 | NOVOGRAD | 0.0718852   |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  97.2914 |  94.1407 | ADAMW    | 1.79616e-06 | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  97.2914 |  97.3235 | NOVOGRAD | 0.820002    |               | final_validation_accuracy |
| ADAGRAD       | NOVOGRAD      |  94.1407 |  97.3235 | NOVOGRAD | 2.94807e-06 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9731 | 0.0017 | 0.0008 | 0.9710 | 0.9752 |
| MILO_LW | 0.9680 | 0.0013 | 0.0006 | 0.9664 | 0.9697 |
| SGD | 0.9711 | 0.0009 | 0.0004 | 0.9700 | 0.9721 |
| ADAMW | 0.9728 | 0.0024 | 0.0011 | 0.9699 | 0.9757 |
| ADAGRAD | 0.9411 | 0.0039 | 0.0017 | 0.9362 | 0.9460 |
| NOVOGRAD | 0.9731 | 0.0021 | 0.0009 | 0.9706 | 0.9757 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.973116 | 0.968024 | MILO     | 0.000910593 | ***           | final_validation_f1_score |
| MILO          | SGD           | 0.973116 | 0.971065 | MILO     | 0.0532615   |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.973116 | 0.9728   | MILO     | 0.814991    |               | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.973116 | 0.941102 | MILO     | 6.55427e-06 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.973116 | 0.973147 | NOVOGRAD | 0.980252    |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.968024 | 0.971065 | SGD      | 0.00401571  | **            | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.968024 | 0.9728   | ADAMW    | 0.00694322  | **            | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.968024 | 0.941102 | MILO_LW  | 3.03333e-05 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.968024 | 0.973147 | NOVOGRAD | 0.00248226  | **            | final_validation_f1_score |
| SGD           | ADAMW         | 0.971065 | 0.9728   | ADAMW    | 0.183324    |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.971065 | 0.941102 | SGD      | 3.85788e-05 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.971065 | 0.973147 | NOVOGRAD | 0.0887216   |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.9728   | 0.941102 | ADAMW    | 1.98429e-06 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.9728   | 0.973147 | NOVOGRAD | 0.811459    |               | final_validation_f1_score |
| ADAGRAD       | NOVOGRAD      | 0.941102 | 0.973147 | NOVOGRAD | 3.13313e-06 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9993 | 0.0001 | 0.0000 | 0.9992 | 0.9994 |
| MILO_LW | 0.9990 | 0.0000 | 0.0000 | 0.9989 | 0.9990 |
| SGD | 0.9992 | 0.0001 | 0.0000 | 0.9991 | 0.9993 |
| ADAMW | 0.9994 | 0.0001 | 0.0000 | 0.9993 | 0.9995 |
| ADAGRAD | 0.9966 | 0.0004 | 0.0002 | 0.9962 | 0.9971 |
| NOVOGRAD | 0.9995 | 0.0000 | 0.0000 | 0.9994 | 0.9995 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.999313 | 0.998967 | MILO     | 9.0835e-06  | ***           | final_validation_auc |
| MILO          | SGD           | 0.999313 | 0.999189 | MILO     | 0.00919049  | **            | final_validation_auc |
| MILO          | ADAMW         | 0.999313 | 0.999435 | ADAMW    | 0.0252664   | *             | final_validation_auc |
| MILO          | ADAGRAD       | 0.999313 | 0.996633 | MILO     | 6.97998e-05 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.999313 | 0.999461 | NOVOGRAD | 0.00219091  | **            | final_validation_auc |
| MILO_LW       | SGD           | 0.998967 | 0.999189 | SGD      | 0.000180649 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.998967 | 0.999435 | ADAMW    | 1.3569e-05  | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.998967 | 0.996633 | MILO_LW  | 0.000132324 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.998967 | 0.999461 | NOVOGRAD | 1.81086e-07 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.999189 | 0.999435 | ADAMW    | 0.000665675 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.999189 | 0.996633 | SGD      | 8.58979e-05 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.999189 | 0.999461 | NOVOGRAD | 4.01552e-05 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.999435 | 0.996633 | ADAMW    | 4.60985e-05 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.999435 | 0.999461 | NOVOGRAD | 0.539381    |               | final_validation_auc |
| ADAGRAD       | NOVOGRAD      | 0.996633 | 0.999461 | NOVOGRAD | 6.27729e-05 | ***           | final_validation_auc |

