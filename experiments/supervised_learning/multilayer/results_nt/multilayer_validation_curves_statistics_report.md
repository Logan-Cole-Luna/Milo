# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.0954 | 0.0043 | 0.0019 | 0.0901 | 0.1008 |
| MILO_LW | 0.0999 | 0.0043 | 0.0019 | 0.0946 | 0.1052 |
| SGD | 0.1009 | 0.0031 | 0.0014 | 0.0971 | 0.1047 |
| ADAMW | 0.2896 | 0.0096 | 0.0043 | 0.2777 | 0.3015 |
| ADAGRAD | 0.1052 | 0.0013 | 0.0006 | 0.1036 | 0.1069 |
| ADEMAMIX | 0.3089 | 0.0425 | 0.0190 | 0.2562 | 0.3616 |
| SOAP | 4395.5817 | 9741.3214 | 4356.4514 | -7699.8664 | 16491.0298 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |       Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|----------:|-------------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.0954293 |    0.0998736 | MILO     | 0.139998    |               | final_validation_loss |
| MILO          | SGD           | 0.0954293 |    0.100881  | MILO     | 0.053071    |               | final_validation_loss |
| MILO          | ADAMW         | 0.0954293 |    0.289603  | MILO     | 4.22219e-08 | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.0954293 |    0.105225  | MILO     | 0.00518658  | **            | final_validation_loss |
| MILO          | ADEMAMIX      | 0.0954293 |    0.308908  | MILO     | 0.000325923 | ***           | final_validation_loss |
| MILO          | SOAP          | 0.0954293 | 4395.58      | MILO     | 0.370071    |               | final_validation_loss |
| MILO_LW       | SGD           | 0.0998736 |    0.100881  | MILO_LW  | 0.681801    |               | final_validation_loss |
| MILO_LW       | ADAMW         | 0.0998736 |    0.289603  | MILO_LW  | 4.83426e-08 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.0998736 |    0.105225  | MILO_LW  | 0.046771    | *             | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.0998736 |    0.308908  | MILO_LW  | 0.000354405 | ***           | final_validation_loss |
| MILO_LW       | SOAP          | 0.0998736 | 4395.58      | MILO_LW  | 0.370072    |               | final_validation_loss |
| SGD           | ADAMW         | 0.100881  |    0.289603  | SGD      | 2.38373e-07 | ***           | final_validation_loss |
| SGD           | ADAGRAD       | 0.100881  |    0.105225  | SGD      | 0.0307076   | *             | final_validation_loss |
| SGD           | ADEMAMIX      | 0.100881  |    0.308908  | SGD      | 0.000377027 | ***           | final_validation_loss |
| SGD           | SOAP          | 0.100881  | 4395.58      | SGD      | 0.370072    |               | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.289603  |    0.105225  | ADAGRAD  | 1.21439e-06 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.289603  |    0.308908  | ADAMW    | 0.372748    |               | final_validation_loss |
| ADAMW         | SOAP          | 0.289603  | 4395.58      | ADAMW    | 0.37009     |               | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 0.105225  |    0.308908  | ADAGRAD  | 0.000424816 | ***           | final_validation_loss |
| ADAGRAD       | SOAP          | 0.105225  | 4395.58      | ADAGRAD  | 0.370072    |               | final_validation_loss |
| ADEMAMIX      | SOAP          | 0.308908  | 4395.58      | ADEMAMIX | 0.370092    |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 97.4074 | 0.1104 | 0.0494 | 97.2703 | 97.5445 |
| MILO_LW | 97.4593 | 0.0738 | 0.0330 | 97.3677 | 97.5509 |
| SGD | 97.0025 | 0.1749 | 0.0782 | 96.7853 | 97.2196 |
| ADAMW | 92.6691 | 0.4303 | 0.1924 | 92.1349 | 93.2034 |
| ADAGRAD | 96.8000 | 0.0816 | 0.0365 | 96.6987 | 96.9013 |
| ADEMAMIX | 91.5580 | 1.1497 | 0.5142 | 90.1305 | 92.9856 |
| SOAP | 82.6593 | 24.1791 | 10.8132 | 52.6369 | 112.6816 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  97.4074 |  97.4593 | MILO_LW  | 0.411644    |               | final_validation_accuracy |
| MILO          | SGD           |  97.4074 |  97.0025 | MILO     | 0.00353601  | **            | final_validation_accuracy |
| MILO          | ADAMW         |  97.4074 |  92.6691 | MILO     | 6.2259e-06  | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  97.4074 |  96.8    | MILO     | 1.63253e-05 | ***           | final_validation_accuracy |
| MILO          | ADEMAMIX      |  97.4074 |  91.558  | MILO     | 0.000313406 | ***           | final_validation_accuracy |
| MILO          | SOAP          |  97.4074 |  82.6593 | MILO     | 0.244307    |               | final_validation_accuracy |
| MILO_LW       | SGD           |  97.4593 |  97.0025 | MILO_LW  | 0.00238699  | **            | final_validation_accuracy |
| MILO_LW       | ADAMW         |  97.4593 |  92.6691 | MILO_LW  | 9.98514e-06 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  97.4593 |  96.8    | MILO_LW  | 1.01048e-06 | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  97.4593 |  91.558  | MILO_LW  | 0.000316882 | ***           | final_validation_accuracy |
| MILO_LW       | SOAP          |  97.4593 |  82.6593 | MILO_LW  | 0.242926    |               | final_validation_accuracy |
| SGD           | ADAMW         |  97.0025 |  92.6691 | SGD      | 2.7835e-06  | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  97.0025 |  96.8    | SGD      | 0.059911    |               | final_validation_accuracy |
| SGD           | ADEMAMIX      |  97.0025 |  91.558  | SGD      | 0.000371104 | ***           | final_validation_accuracy |
| SGD           | SOAP          |  97.0025 |  82.6593 | SGD      | 0.255358    |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  92.6691 |  96.8    | ADAGRAD  | 1.70402e-05 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  92.6691 |  91.558  | ADAMW    | 0.0977787   |               | final_validation_accuracy |
| ADAMW         | SOAP          |  92.6691 |  82.6593 | ADAMW    | 0.407029    |               | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  96.8    |  91.558  | ADAGRAD  | 0.00050038  | ***           | final_validation_accuracy |
| ADAGRAD       | SOAP          |  96.8    |  82.6593 | ADAGRAD  | 0.261062    |               | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  91.558  |  82.6593 | ADEMAMIX | 0.457043    |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9740 | 0.0011 | 0.0005 | 0.9726 | 0.9754 |
| MILO_LW | 0.9745 | 0.0007 | 0.0003 | 0.9737 | 0.9754 |
| SGD | 0.9699 | 0.0018 | 0.0008 | 0.9677 | 0.9721 |
| ADAMW | 0.9268 | 0.0040 | 0.0018 | 0.9219 | 0.9317 |
| ADAGRAD | 0.9679 | 0.0008 | 0.0004 | 0.9669 | 0.9689 |
| ADEMAMIX | 0.9144 | 0.0127 | 0.0057 | 0.8986 | 0.9302 |
| SOAP | 0.8277 | 0.2383 | 0.1066 | 0.5318 | 1.1236 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.974003 | 0.974536 | MILO_LW  | 0.402083    |               | final_validation_f1_score |
| MILO          | SGD           | 0.974003 | 0.969936 | MILO     | 0.00372343  | **            | final_validation_f1_score |
| MILO          | ADAMW         | 0.974003 | 0.92682  | MILO     | 3.43124e-06 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.974003 | 0.967895 | MILO     | 1.87169e-05 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.974003 | 0.91437  | MILO     | 0.000439062 | ***           | final_validation_f1_score |
| MILO          | SOAP          | 0.974003 | 0.827713 | MILO     | 0.241791    |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.974536 | 0.969936 | MILO_LW  | 0.00258253  | **            | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.974536 | 0.92682  | MILO_LW  | 6.82582e-06 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.974536 | 0.967895 | MILO_LW  | 1.00953e-06 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.974536 | 0.91437  | MILO_LW  | 0.000441537 | ***           | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.974536 | 0.827713 | MILO_LW  | 0.240365    |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.969936 | 0.92682  | SGD      | 1.21766e-06 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.969936 | 0.967895 | SGD      | 0.0610204   |               | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.969936 | 0.91437  | SGD      | 0.00053029  | ***           | final_validation_f1_score |
| SGD           | SOAP          | 0.969936 | 0.827713 | SGD      | 0.252943    |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.92682  | 0.967895 | ADAGRAD  | 1.08642e-05 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.92682  | 0.91437  | ADAMW    | 0.0937791   |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.92682  | 0.827713 | ADAMW    | 0.405077    |               | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.967895 | 0.91437  | ADAGRAD  | 0.00069001  | ***           | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.967895 | 0.827713 | ADAGRAD  | 0.258724    |               | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.91437  | 0.827713 | ADEMAMIX | 0.46213     |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9995 | 0.0000 | 0.0000 | 0.9994 | 0.9995 |
| MILO_LW | 0.9995 | 0.0001 | 0.0000 | 0.9994 | 0.9995 |
| SGD | 0.9992 | 0.0001 | 0.0000 | 0.9991 | 0.9993 |
| ADAMW | 0.9947 | 0.0004 | 0.0002 | 0.9943 | 0.9952 |
| ADAGRAD | 0.9991 | 0.0001 | 0.0000 | 0.9991 | 0.9992 |
| ADEMAMIX | 0.9947 | 0.0009 | 0.0004 | 0.9936 | 0.9958 |
| SOAP | 0.9291 | 0.1438 | 0.0643 | 0.7506 | 1.1076 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.999478 | 0.999457 | MILO     | 0.529257    |               | final_validation_auc |
| MILO          | SGD           | 0.999478 | 0.99917  | MILO     | 0.000111659 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.999478 | 0.994743 | MILO     | 9.79718e-06 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.999478 | 0.99914  | MILO     | 4.26459e-05 | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.999478 | 0.994659 | MILO     | 0.000263565 | ***           | final_validation_auc |
| MILO          | SOAP          | 0.999478 | 0.929121 | MILO     | 0.335339    |               | final_validation_auc |
| MILO_LW       | SGD           | 0.999457 | 0.99917  | MILO_LW  | 0.000134233 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.999457 | 0.994743 | MILO_LW  | 7.91817e-06 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.999457 | 0.99914  | MILO_LW  | 5.22963e-05 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.999457 | 0.994659 | MILO_LW  | 0.000260913 | ***           | final_validation_auc |
| MILO_LW       | SOAP          | 0.999457 | 0.929121 | MILO_LW  | 0.335468    |               | final_validation_auc |
| SGD           | ADAMW         | 0.99917  | 0.994743 | SGD      | 9.24146e-06 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.99917  | 0.99914  | SGD      | 0.514892    |               | final_validation_auc |
| SGD           | ADEMAMIX      | 0.99917  | 0.994659 | SGD      | 0.000328543 | ***           | final_validation_auc |
| SGD           | SOAP          | 0.99917  | 0.929121 | SGD      | 0.337214    |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.994743 | 0.99914  | ADAGRAD  | 9.96007e-06 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.994743 | 0.994659 | ADAMW    | 0.853899    |               | final_validation_auc |
| ADAMW         | SOAP          | 0.994743 | 0.929121 | ADAMW    | 0.365166    |               | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.99914  | 0.994659 | ADAGRAD  | 0.000338985 | ***           | final_validation_auc |
| ADAGRAD       | SOAP          | 0.99914  | 0.929121 | ADAGRAD  | 0.337393    |               | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.994659 | 0.929121 | ADEMAMIX | 0.365719    |               | final_validation_auc |

