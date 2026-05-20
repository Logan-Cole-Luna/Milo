# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.2186 | 0.0166 | 0.0074 | 0.1980 | 0.2392 |
| MILO_LW | 0.2181 | 0.0177 | 0.0079 | 0.1961 | 0.2401 |
| SGD | 0.0826 | 0.0028 | 0.0012 | 0.0792 | 0.0860 |
| ADAMW | 0.3098 | 0.0332 | 0.0148 | 0.2686 | 0.3509 |
| ADAM_MINI | 2.3047 | 0.0026 | 0.0011 | 2.3015 | 2.3079 |
| NOVOGRAD | 0.1122 | 0.0083 | 0.0037 | 0.1019 | 0.1225 |
| ADAGRAD | 0.0898 | 0.0033 | 0.0015 | 0.0857 | 0.0939 |
| ADEMAMIX | 0.2948 | 0.0273 | 0.0122 | 0.2610 | 0.3287 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |    Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|----------:|----------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.218572  | 0.218092  | MILO_LW  | 0.965805    |               | final_validation_loss |
| MILO          | SGD           | 0.218572  | 0.0825974 | SGD      | 3.68828e-05 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.218572  | 0.309753  | MILO     | 0.00161341  | **            | final_validation_loss |
| MILO          | ADAM_MINI     | 0.218572  | 2.30473   | MILO     | 4.23264e-10 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      | 0.218572  | 0.112223  | NOVOGRAD | 1.60256e-05 | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.218572  | 0.0897718 | ADAGRAD  | 4.03449e-05 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      | 0.218572  | 0.294847  | MILO     | 0.00130015  | **            | final_validation_loss |
| MILO_LW       | SGD           | 0.218092  | 0.0825974 | SGD      | 5.08531e-05 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         | 0.218092  | 0.309753  | MILO_LW  | 0.00149254  | **            | final_validation_loss |
| MILO_LW       | ADAM_MINI     | 0.218092  | 2.30473   | MILO_LW  | 6.08918e-10 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.218092  | 0.112223  | NOVOGRAD | 2.84004e-05 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.218092  | 0.0897718 | ADAGRAD  | 5.67559e-05 | ***           | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.218092  | 0.294847  | MILO_LW  | 0.00123039  | **            | final_validation_loss |
| SGD           | ADAMW         | 0.0825974 | 0.309753  | SGD      | 9.78513e-05 | ***           | final_validation_loss |
| SGD           | ADAM_MINI     | 0.0825974 | 2.30473   | SGD      | 1.62698e-22 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 0.0825974 | 0.112223  | SGD      | 0.000714919 | ***           | final_validation_loss |
| SGD           | ADAGRAD       | 0.0825974 | 0.0897718 | SGD      | 0.00616382  | **            | final_validation_loss |
| SGD           | ADEMAMIX      | 0.0825974 | 0.294847  | SGD      | 5.65993e-05 | ***           | final_validation_loss |
| ADAMW         | ADAM_MINI     | 0.309753  | 2.30473   | ADAMW    | 1.54171e-08 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.309753  | 0.112223  | NOVOGRAD | 9.94117e-05 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.309753  | 0.0897718 | ADAGRAD  | 0.000107747 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.309753  | 0.294847  | ADEMAMIX | 0.460791    |               | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      | 2.30473   | 0.112223  | NOVOGRAD | 1.15938e-12 | ***           | final_validation_loss |
| ADAM_MINI     | ADAGRAD       | 2.30473   | 0.0897718 | ADAGRAD  | 3.68268e-21 | ***           | final_validation_loss |
| ADAM_MINI     | ADEMAMIX      | 2.30473   | 0.294847  | ADEMAMIX | 6.2488e-09  | ***           | final_validation_loss |
| NOVOGRAD      | ADAGRAD       | 0.112223  | 0.0897718 | ADAGRAD  | 0.00214657  | **            | final_validation_loss |
| NOVOGRAD      | ADEMAMIX      | 0.112223  | 0.294847  | NOVOGRAD | 4.44142e-05 | ***           | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 0.0897718 | 0.294847  | ADAGRAD  | 6.17941e-05 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 97.3901 | 0.1782 | 0.0797 | 97.1689 | 97.6114 |
| MILO_LW | 97.3802 | 0.1796 | 0.0803 | 97.1572 | 97.6033 |
| SGD | 97.4444 | 0.0746 | 0.0334 | 97.3518 | 97.5371 |
| ADAMW | 92.7259 | 0.7206 | 0.3223 | 91.8311 | 93.6207 |
| ADAM_MINI | 10.1160 | 0.1837 | 0.0821 | 9.8880 | 10.3441 |
| NOVOGRAD | 96.6741 | 0.3213 | 0.1437 | 96.2751 | 97.0730 |
| ADAGRAD | 97.1802 | 0.0955 | 0.0427 | 97.0617 | 97.2988 |
| ADEMAMIX | 92.2247 | 0.9134 | 0.4085 | 91.0906 | 93.3588 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  97.3901 |  97.3802 | MILO     | 0.932589    |               | final_validation_accuracy |
| MILO          | SGD           |  97.3901 |  97.4444 | SGD      | 0.555289    |               | final_validation_accuracy |
| MILO          | ADAMW         |  97.3901 |  92.7259 | MILO     | 7.02024e-05 | ***           | final_validation_accuracy |
| MILO          | ADAM_MINI     |  97.3901 |  10.116  | MILO     | 1.01625e-20 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  97.3901 |  96.6741 | MILO     | 0.00434655  | **            | final_validation_accuracy |
| MILO          | ADAGRAD       |  97.3901 |  97.1802 | MILO     | 0.0584717   |               | final_validation_accuracy |
| MILO          | ADEMAMIX      |  97.3901 |  92.2247 | MILO     | 0.000156523 | ***           | final_validation_accuracy |
| MILO_LW       | SGD           |  97.3802 |  97.4444 | SGD      | 0.491628    |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  97.3802 |  92.7259 | MILO_LW  | 7.01671e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  97.3802 |  10.116  | MILO_LW  | 1.03156e-20 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  97.3802 |  96.6741 | MILO_LW  | 0.00464204  | **            | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  97.3802 |  97.1802 | MILO_LW  | 0.0695705   |               | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  97.3802 |  92.2247 | MILO_LW  | 0.000156885 | ***           | final_validation_accuracy |
| SGD           | ADAMW         |  97.4444 |  92.7259 | SGD      | 0.000112672 | ***           | final_validation_accuracy |
| SGD           | ADAM_MINI     |  97.4444 |  10.116  | SGD      | 4.08523e-15 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  97.4444 |  96.6741 | SGD      | 0.0048242   | **            | final_validation_accuracy |
| SGD           | ADAGRAD       |  97.4444 |  97.1802 | SGD      | 0.00144809  | **            | final_validation_accuracy |
| SGD           | ADEMAMIX      |  97.4444 |  92.2247 | SGD      | 0.00020235  | ***           | final_validation_accuracy |
| ADAMW         | ADAM_MINI     |  92.7259 |  10.116  | ADAMW    | 1.62159e-10 | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  92.7259 |  96.6741 | NOVOGRAD | 5.24689e-05 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  92.7259 |  97.1802 | ADAGRAD  | 0.000132352 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  92.7259 |  92.2247 | ADAMW    | 0.36506     |               | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  10.116  |  96.6741 | NOVOGRAD | 5.55621e-16 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  10.116  |  97.1802 | ADAGRAD  | 8.9915e-17  | ***           | final_validation_accuracy |
| ADAM_MINI     | ADEMAMIX      |  10.116  |  92.2247 | ADEMAMIX | 1.03243e-09 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADAGRAD       |  96.6741 |  97.1802 | ADAGRAD  | 0.0217175   | *             | final_validation_accuracy |
| NOVOGRAD      | ADEMAMIX      |  96.6741 |  92.2247 | NOVOGRAD | 0.000154461 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  97.1802 |  92.2247 | ADAGRAD  | 0.000238889 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9738 | 0.0018 | 0.0008 | 0.9716 | 0.9760 |
| MILO_LW | 0.9737 | 0.0018 | 0.0008 | 0.9714 | 0.9760 |
| SGD | 0.9743 | 0.0007 | 0.0003 | 0.9734 | 0.9752 |
| ADAMW | 0.9274 | 0.0067 | 0.0030 | 0.9190 | 0.9358 |
| ADAM_MINI | 0.0191 | 0.0007 | 0.0003 | 0.0181 | 0.0200 |
| NOVOGRAD | 0.9666 | 0.0033 | 0.0015 | 0.9625 | 0.9707 |
| ADAGRAD | 0.9717 | 0.0009 | 0.0004 | 0.9706 | 0.9729 |
| ADEMAMIX | 0.9221 | 0.0089 | 0.0040 | 0.9109 | 0.9332 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |    Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|----------:|----------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.973762  | 0.973717  | MILO     | 0.969142    |               | final_validation_f1_score |
| MILO          | SGD           | 0.973762  | 0.974322  | SGD      | 0.541483    |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.973762  | 0.927395  | MILO     | 4.95068e-05 | ***           | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.973762  | 0.0190589 | MILO     | 1.71221e-15 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.973762  | 0.966589  | MILO     | 0.00489554  | **            | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.973762  | 0.971737  | MILO     | 0.0643839   |               | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.973762  | 0.922051  | MILO     | 0.000140841 | ***           | final_validation_f1_score |
| MILO_LW       | SGD           | 0.973717  | 0.974322  | SGD      | 0.521766    |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.973717  | 0.927395  | MILO_LW  | 4.74933e-05 | ***           | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.973717  | 0.0190589 | MILO_LW  | 3.08106e-15 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.973717  | 0.966589  | MILO_LW  | 0.0050095   | **            | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.973717  | 0.971737  | MILO_LW  | 0.0752924   |               | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.973717  | 0.922051  | MILO_LW  | 0.000137996 | ***           | final_validation_f1_score |
| SGD           | ADAMW         | 0.974322  | 0.927395  | SGD      | 8.70063e-05 | ***           | final_validation_f1_score |
| SGD           | ADAM_MINI     | 0.974322  | 0.0190589 | SGD      | 3.48914e-24 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.974322  | 0.966589  | SGD      | 0.00530036  | **            | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.974322  | 0.971737  | SGD      | 0.00150992  | **            | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.974322  | 0.922051  | SGD      | 0.000184852 | ***           | final_validation_f1_score |
| ADAMW         | ADAM_MINI     | 0.927395  | 0.0190589 | ADAMW    | 4.83221e-10 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.927395  | 0.966589  | NOVOGRAD | 2.99344e-05 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.927395  | 0.971737  | ADAGRAD  | 0.000100834 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.927395  | 0.922051  | ADAMW    | 0.319435    |               | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.0190589 | 0.966589  | NOVOGRAD | 4.71879e-12 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.0190589 | 0.971737  | ADAGRAD  | 1.65725e-22 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADEMAMIX      | 0.0190589 | 0.922051  | ADEMAMIX | 1.85531e-09 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADAGRAD       | 0.966589  | 0.971737  | ADAGRAD  | 0.0224658   | *             | final_validation_f1_score |
| NOVOGRAD      | ADEMAMIX      | 0.966589  | 0.922051  | NOVOGRAD | 0.000128185 | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.971737  | 0.922051  | ADAGRAD  | 0.000216999 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9993 | 0.0001 | 0.0000 | 0.9992 | 0.9994 |
| MILO_LW | 0.9993 | 0.0001 | 0.0000 | 0.9992 | 0.9995 |
| SGD | 0.9995 | 0.0000 | 0.0000 | 0.9994 | 0.9995 |
| ADAMW | 0.9943 | 0.0013 | 0.0006 | 0.9926 | 0.9959 |
| ADAM_MINI | 0.5005 | 0.0005 | 0.0002 | 0.4999 | 0.5011 |
| NOVOGRAD | 0.9992 | 0.0000 | 0.0000 | 0.9992 | 0.9993 |
| ADAGRAD | 0.9994 | 0.0001 | 0.0000 | 0.9993 | 0.9994 |
| ADEMAMIX | 0.9950 | 0.0006 | 0.0002 | 0.9943 | 0.9956 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.999314 | 0.999346 | MILO_LW  | 0.569364    |               | final_validation_auc |
| MILO          | SGD           | 0.999314 | 0.999465 | SGD      | 0.00570341  | **            | final_validation_auc |
| MILO          | ADAMW         | 0.999314 | 0.994261 | MILO     | 0.000974048 | ***           | final_validation_auc |
| MILO          | ADAM_MINI     | 0.999314 | 0.500463 | MILO     | 6.1329e-14  | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.999314 | 0.999234 | MILO     | 0.0668562   |               | final_validation_auc |
| MILO          | ADAGRAD       | 0.999314 | 0.999359 | ADAGRAD  | 0.294833    |               | final_validation_auc |
| MILO          | ADEMAMIX      | 0.999314 | 0.99496  | MILO     | 5.03655e-05 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.999346 | 0.999465 | SGD      | 0.0483466   | *             | final_validation_auc |
| MILO_LW       | ADAMW         | 0.999346 | 0.994261 | MILO_LW  | 0.000934785 | ***           | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.999346 | 0.500463 | MILO_LW  | 2.42517e-14 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.999346 | 0.999234 | MILO_LW  | 0.0583987   |               | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.999346 | 0.999359 | ADAGRAD  | 0.803262    |               | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.999346 | 0.99496  | MILO_LW  | 4.15839e-05 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.999465 | 0.994261 | SGD      | 0.000881086 | ***           | final_validation_auc |
| SGD           | ADAM_MINI     | 0.999465 | 0.500463 | SGD      | 1.26082e-13 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.999465 | 0.999234 | SGD      | 2.70243e-05 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.999465 | 0.999359 | SGD      | 0.00855646  | **            | final_validation_auc |
| SGD           | ADEMAMIX      | 0.999465 | 0.99496  | SGD      | 4.98389e-05 | ***           | final_validation_auc |
| ADAMW         | ADAM_MINI     | 0.994261 | 0.500463 | ADAMW    | 5.05244e-14 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.994261 | 0.999234 | NOVOGRAD | 0.00105349  | **            | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.994261 | 0.999359 | ADAGRAD  | 0.000951112 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.994261 | 0.99496  | ADEMAMIX | 0.318538    |               | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.500463 | 0.999234 | NOVOGRAD | 1.61736e-13 | ***           | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.500463 | 0.999359 | ADAGRAD  | 1.07073e-13 | ***           | final_validation_auc |
| ADAM_MINI     | ADEMAMIX      | 0.500463 | 0.99496  | ADEMAMIX | 1.18682e-22 | ***           | final_validation_auc |
| NOVOGRAD      | ADAGRAD       | 0.999234 | 0.999359 | ADAGRAD  | 0.00298122  | **            | final_validation_auc |
| NOVOGRAD      | ADEMAMIX      | 0.999234 | 0.99496  | NOVOGRAD | 6.42471e-05 | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.999359 | 0.99496  | ADAGRAD  | 5.32851e-05 | ***           | final_validation_auc |

