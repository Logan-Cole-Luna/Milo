# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 2.0792 | 0.0206 | 0.0092 | 2.0537 | 2.1048 |
| MILO_LW | 2.1189 | 0.0345 | 0.0154 | 2.0761 | 2.1618 |
| SGD | 2.7783 | 0.1141 | 0.0510 | 2.6366 | 2.9200 |
| ADAMW | 2.4744 | 0.0921 | 0.0412 | 2.3600 | 2.5887 |
| ADAM_MINI | 3.6156 | 0.0178 | 0.0080 | 3.5935 | 3.6378 |
| NOVOGRAD | 2.7037 | 0.0392 | 0.0175 | 2.6550 | 2.7523 |
| ADAGRAD | 2.5361 | 0.0358 | 0.0160 | 2.4917 | 2.5806 |
| ADEMAMIX | 2.5718 | 0.0810 | 0.0362 | 2.4712 | 2.6724 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  2.07923 |  2.11894 | MILO     | 0.0653894   |               | final_validation_loss |
| MILO          | SGD           |  2.07923 |  2.77832 | MILO     | 0.000118165 | ***           | final_validation_loss |
| MILO          | ADAMW         |  2.07923 |  2.47436 | MILO     | 0.000455094 | ***           | final_validation_loss |
| MILO          | ADAM_MINI     |  2.07923 |  3.61564 | MILO     | 2.99098e-14 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      |  2.07923 |  2.70365 | MILO     | 5.99841e-08 | ***           | final_validation_loss |
| MILO          | ADAGRAD       |  2.07923 |  2.53612 | MILO     | 1.39079e-07 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      |  2.07923 |  2.57179 | MILO     | 8.94799e-05 | ***           | final_validation_loss |
| MILO_LW       | SGD           |  2.11894 |  2.77832 | MILO_LW  | 8.88633e-05 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         |  2.11894 |  2.47436 | MILO_LW  | 0.000427988 | ***           | final_validation_loss |
| MILO_LW       | ADAM_MINI     |  2.11894 |  3.61564 | MILO_LW  | 1.65726e-10 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      |  2.11894 |  2.70365 | MILO_LW  | 8.60261e-09 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       |  2.11894 |  2.53612 | MILO_LW  | 6.81949e-08 | ***           | final_validation_loss |
| MILO_LW       | ADEMAMIX      |  2.11894 |  2.57179 | MILO_LW  | 5.28682e-05 | ***           | final_validation_loss |
| SGD           | ADAMW         |  2.77832 |  2.47436 | ADAMW    | 0.00188447  | **            | final_validation_loss |
| SGD           | ADAM_MINI     |  2.77832 |  3.61564 | SGD      | 6.08661e-05 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      |  2.77832 |  2.70365 | NOVOGRAD | 0.225901    |               | final_validation_loss |
| SGD           | ADAGRAD       |  2.77832 |  2.53612 | ADAGRAD  | 0.00696172  | **            | final_validation_loss |
| SGD           | ADEMAMIX      |  2.77832 |  2.57179 | ADEMAMIX | 0.0125819   | *             | final_validation_loss |
| ADAMW         | ADAM_MINI     |  2.47436 |  3.61564 | ADAMW    | 5.61139e-06 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      |  2.47436 |  2.70365 | ADAMW    | 0.00295881  | **            | final_validation_loss |
| ADAMW         | ADAGRAD       |  2.47436 |  2.53612 | ADAMW    | 0.219088    |               | final_validation_loss |
| ADAMW         | ADEMAMIX      |  2.47436 |  2.57179 | ADAMW    | 0.114212    |               | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      |  3.61564 |  2.70365 | NOVOGRAD | 1.69501e-08 | ***           | final_validation_loss |
| ADAM_MINI     | ADAGRAD       |  3.61564 |  2.53612 | ADAGRAD  | 2.003e-09   | ***           | final_validation_loss |
| ADAM_MINI     | ADEMAMIX      |  3.61564 |  2.57179 | ADEMAMIX | 4.00808e-06 | ***           | final_validation_loss |
| NOVOGRAD      | ADAGRAD       |  2.70365 |  2.53612 | ADAGRAD  | 0.000110358 | ***           | final_validation_loss |
| NOVOGRAD      | ADEMAMIX      |  2.70365 |  2.57179 | ADEMAMIX | 0.0178663   | *             | final_validation_loss |
| ADAGRAD       | ADEMAMIX      |  2.53612 |  2.57179 | ADAGRAD  | 0.405519    |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 47.5970 | 0.6598 | 0.2951 | 46.7777 | 48.4163 |
| MILO_LW | 47.1881 | 0.5535 | 0.2475 | 46.5009 | 47.8754 |
| SGD | 36.5570 | 1.4371 | 0.6427 | 34.7726 | 38.3415 |
| ADAMW | 47.2533 | 1.3837 | 0.6188 | 45.5353 | 48.9714 |
| ADAM_MINI | 15.3126 | 0.2358 | 0.1054 | 15.0198 | 15.6054 |
| NOVOGRAD | 32.6785 | 0.5888 | 0.2633 | 31.9474 | 33.4096 |
| ADAGRAD | 40.7556 | 0.4140 | 0.1852 | 40.2415 | 41.2696 |
| ADEMAMIX | 48.1511 | 1.5022 | 0.6718 | 46.2859 | 50.0163 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  47.597  |  47.1881 | MILO     | 0.320303    |               | final_validation_accuracy |
| MILO          | SGD           |  47.597  |  36.557  | MILO     | 7.70657e-06 | ***           | final_validation_accuracy |
| MILO          | ADAMW         |  47.597  |  47.2533 | MILO     | 0.634799    |               | final_validation_accuracy |
| MILO          | ADAM_MINI     |  47.597  |  15.3126 | MILO     | 1.60505e-09 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  47.597  |  32.6785 | MILO     | 3.3338e-10  | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  47.597  |  40.7556 | MILO     | 3.4292e-07  | ***           | final_validation_accuracy |
| MILO          | ADEMAMIX      |  47.597  |  48.1511 | ADEMAMIX | 0.481303    |               | final_validation_accuracy |
| MILO_LW       | SGD           |  47.1881 |  36.557  | MILO_LW  | 1.61875e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADAMW         |  47.1881 |  47.2533 | ADAMW    | 0.925712    |               | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  47.1881 |  15.3126 | MILO_LW  | 1.93485e-10 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  47.1881 |  32.6785 | MILO_LW  | 1.74359e-10 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  47.1881 |  40.7556 | MILO_LW  | 7.63639e-08 | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  47.1881 |  48.1511 | ADEMAMIX | 0.235682    |               | final_validation_accuracy |
| SGD           | ADAMW         |  36.557  |  47.2533 | ADAMW    | 2.18525e-06 | ***           | final_validation_accuracy |
| SGD           | ADAM_MINI     |  36.557  |  15.3126 | SGD      | 3.14987e-06 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  36.557  |  32.6785 | SGD      | 0.00209891  | **            | final_validation_accuracy |
| SGD           | ADAGRAD       |  36.557  |  40.7556 | ADAGRAD  | 0.00194193  | **            | final_validation_accuracy |
| SGD           | ADEMAMIX      |  36.557  |  48.1511 | ADEMAMIX | 1.62578e-06 | ***           | final_validation_accuracy |
| ADAMW         | ADAM_MINI     |  47.2533 |  15.3126 | ADAMW    | 4.62792e-07 | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  47.2533 |  32.6785 | ADAMW    | 1.84309e-06 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  47.2533 |  40.7556 | ADAMW    | 0.000231948 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  47.2533 |  48.1511 | ADEMAMIX | 0.35461     |               | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  15.3126 |  32.6785 | NOVOGRAD | 1.06737e-08 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  15.3126 |  40.7556 | ADAGRAD  | 7.01609e-12 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADEMAMIX      |  15.3126 |  48.1511 | ADEMAMIX | 6.36119e-07 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADAGRAD       |  32.6785 |  40.7556 | ADAGRAD  | 2.94651e-08 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADEMAMIX      |  32.6785 |  48.1511 | ADEMAMIX | 2.82135e-06 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  40.7556 |  48.1511 | ADEMAMIX | 0.00020721  | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.4724 | 0.0080 | 0.0036 | 0.4624 | 0.4824 |
| MILO_LW | 0.4685 | 0.0048 | 0.0021 | 0.4625 | 0.4744 |
| SGD | 0.3627 | 0.0162 | 0.0072 | 0.3426 | 0.3827 |
| ADAMW | 0.4696 | 0.0118 | 0.0053 | 0.4549 | 0.4843 |
| ADAM_MINI | 0.1294 | 0.0042 | 0.0019 | 0.1242 | 0.1346 |
| NOVOGRAD | 0.3190 | 0.0058 | 0.0026 | 0.3118 | 0.3261 |
| ADAGRAD | 0.4088 | 0.0038 | 0.0017 | 0.4041 | 0.4135 |
| ADEMAMIX | 0.4830 | 0.0141 | 0.0063 | 0.4654 | 0.5005 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.472411 | 0.468459 | MILO     | 0.379058    |               | final_validation_f1_score |
| MILO          | SGD           | 0.472411 | 0.362651 | MILO     | 1.17093e-05 | ***           | final_validation_f1_score |
| MILO          | ADAMW         | 0.472411 | 0.46958  | MILO     | 0.671795    |               | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.472411 | 0.129373 | MILO     | 1.66545e-10 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.472411 | 0.31895  | MILO     | 2.51765e-09 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.472411 | 0.40881  | MILO     | 6.09828e-06 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.472411 | 0.482968 | ADEMAMIX | 0.193775    |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.468459 | 0.362651 | MILO_LW  | 5.14025e-05 | ***           | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.468459 | 0.46958  | ADAMW    | 0.851787    |               | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.468459 | 0.129373 | MILO_LW  | 4.33981e-14 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.468459 | 0.31895  | MILO_LW  | 1.29338e-10 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.468459 | 0.40881  | MILO_LW  | 3.99463e-08 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.468459 | 0.482968 | ADEMAMIX | 0.0824729   |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.362651 | 0.46958  | ADAMW    | 4.5573e-06  | ***           | final_validation_f1_score |
| SGD           | ADAM_MINI     | 0.362651 | 0.129373 | SGD      | 1.78763e-06 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.362651 | 0.31895  | SGD      | 0.00232501  | **            | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.362651 | 0.40881  | ADAGRAD  | 0.00239849  | **            | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.362651 | 0.482968 | ADEMAMIX | 1.79249e-06 | ***           | final_validation_f1_score |
| ADAMW         | ADAM_MINI     | 0.46958  | 0.129373 | ADAMW    | 2.38365e-08 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.46958  | 0.31895  | ADAMW    | 3.51285e-07 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.46958  | 0.40881  | ADAMW    | 0.000141742 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.46958  | 0.482968 | ADEMAMIX | 0.144057    |               | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.129373 | 0.31895  | NOVOGRAD | 4.33402e-11 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.129373 | 0.40881  | ADAGRAD  | 6.72268e-14 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADEMAMIX      | 0.129373 | 0.482968 | ADEMAMIX | 9.66041e-08 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADAGRAD       | 0.31895  | 0.40881  | ADAGRAD  | 1.77009e-08 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADEMAMIX      | 0.31895  | 0.482968 | ADEMAMIX | 1.28795e-06 | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.40881  | 0.482968 | ADEMAMIX | 0.000160766 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9556 | 0.0010 | 0.0004 | 0.9544 | 0.9568 |
| MILO_LW | 0.9526 | 0.0025 | 0.0011 | 0.9495 | 0.9557 |
| SGD | 0.9361 | 0.0045 | 0.0020 | 0.9305 | 0.9416 |
| ADAMW | 0.9656 | 0.0020 | 0.0009 | 0.9631 | 0.9681 |
| ADAM_MINI | 0.8438 | 0.0028 | 0.0013 | 0.8403 | 0.8473 |
| NOVOGRAD | 0.9297 | 0.0010 | 0.0005 | 0.9284 | 0.9310 |
| ADAGRAD | 0.9385 | 0.0013 | 0.0006 | 0.9369 | 0.9402 |
| ADEMAMIX | 0.9658 | 0.0029 | 0.0013 | 0.9622 | 0.9694 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.955644 | 0.952589 | MILO     | 0.0493527   | *             | final_validation_auc |
| MILO          | SGD           | 0.955644 | 0.936067 | MILO     | 0.000427644 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.955644 | 0.965574 | ADAMW    | 8.03747e-05 | ***           | final_validation_auc |
| MILO          | ADAM_MINI     | 0.955644 | 0.843827 | MILO     | 5.46728e-09 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.955644 | 0.929689 | MILO     | 1.43038e-10 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.955644 | 0.938532 | MILO     | 4.03433e-08 | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.955644 | 0.965762 | ADEMAMIX | 0.00078399  | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.952589 | 0.936067 | MILO_LW  | 0.000294196 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.952589 | 0.965574 | ADAMW    | 2.33272e-05 | ***           | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.952589 | 0.843827 | MILO_LW  | 4.82519e-12 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.952589 | 0.929689 | MILO_LW  | 4.30282e-06 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.952589 | 0.938532 | MILO_LW  | 2.78837e-05 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.952589 | 0.965762 | ADEMAMIX | 6.43732e-05 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.936067 | 0.965574 | ADAMW    | 1.85487e-05 | ***           | final_validation_auc |
| SGD           | ADAM_MINI     | 0.936067 | 0.843827 | SGD      | 3.4773e-09  | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.936067 | 0.929689 | SGD      | 0.0313845   | *             | final_validation_auc |
| SGD           | ADAGRAD       | 0.936067 | 0.938532 | ADAGRAD  | 0.293761    |               | final_validation_auc |
| SGD           | ADEMAMIX      | 0.936067 | 0.965762 | ADEMAMIX | 5.83539e-06 | ***           | final_validation_auc |
| ADAMW         | ADAM_MINI     | 0.965574 | 0.843827 | ADAMW    | 6.30818e-12 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.965574 | 0.929689 | ADAMW    | 3.96935e-08 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.965574 | 0.938532 | ADAMW    | 4.92723e-08 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.965574 | 0.965762 | ADEMAMIX | 0.908584    |               | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.843827 | 0.929689 | NOVOGRAD | 1.4872e-08  | ***           | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.843827 | 0.938532 | ADAGRAD  | 1.55892e-09 | ***           | final_validation_auc |
| ADAM_MINI     | ADEMAMIX      | 0.843827 | 0.965762 | ADEMAMIX | 2.61595e-12 | ***           | final_validation_auc |
| NOVOGRAD      | ADAGRAD       | 0.929689 | 0.938532 | ADAGRAD  | 4.13629e-06 | ***           | final_validation_auc |
| NOVOGRAD      | ADEMAMIX      | 0.929689 | 0.965762 | ADEMAMIX | 1.53919e-06 | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.938532 | 0.965762 | ADEMAMIX | 2.53475e-06 | ***           | final_validation_auc |

