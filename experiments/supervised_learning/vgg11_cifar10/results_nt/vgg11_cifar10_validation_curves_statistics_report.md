# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.8720 | 0.0418 | 0.0187 | 0.8201 | 0.9238 |
| MILO_LW | 0.9152 | 0.0677 | 0.0303 | 0.8311 | 0.9992 |
| SGD | 1.2035 | 0.0308 | 0.0138 | 1.1653 | 1.2418 |
| ADAMW | 0.6494 | 0.0246 | 0.0110 | 0.6189 | 0.6799 |
| ADAM_MINI | 1.7865 | 0.0502 | 0.0225 | 1.7242 | 1.8489 |
| NOVOGRAD | 1.2071 | 0.0374 | 0.0167 | 1.1607 | 1.2535 |
| ADAGRAD | 1.4605 | 0.1051 | 0.0470 | 1.3299 | 1.5910 |
| ADEMAMIX | 0.6826 | 0.0140 | 0.0063 | 0.6652 | 0.7000 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.871959 | 0.915154 | MILO     | 0.266008    |               | final_validation_loss |
| MILO          | SGD           | 0.871959 | 1.20353  | MILO     | 1.23847e-06 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.871959 | 0.649415 | ADAMW    | 3.03038e-05 | ***           | final_validation_loss |
| MILO          | ADAM_MINI     | 0.871959 | 1.78655  | MILO     | 1.95592e-09 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      | 0.871959 | 1.20712  | MILO     | 1.04388e-06 | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.871959 | 1.46047  | MILO     | 6.15688e-05 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      | 0.871959 | 0.682571 | ADEMAMIX | 0.000232632 | ***           | final_validation_loss |
| MILO_LW       | SGD           | 0.915154 | 1.20353  | MILO_LW  | 0.000190673 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         | 0.915154 | 0.649415 | ADAMW    | 0.000412246 | ***           | final_validation_loss |
| MILO_LW       | ADAM_MINI     | 0.915154 | 1.78655  | MILO_LW  | 3.74411e-08 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.915154 | 1.20712  | MILO_LW  | 0.000123428 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.915154 | 1.46047  | MILO_LW  | 2.95886e-05 | ***           | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.915154 | 0.682571 | ADEMAMIX | 0.00119953  | **            | final_validation_loss |
| SGD           | ADAMW         | 1.20353  | 0.649415 | ADAMW    | 2.40306e-09 | ***           | final_validation_loss |
| SGD           | ADAM_MINI     | 1.20353  | 1.78655  | SGD      | 1.80944e-07 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 1.20353  | 1.20712  | SGD      | 0.872335    |               | final_validation_loss |
| SGD           | ADAGRAD       | 1.20353  | 1.46047  | SGD      | 0.00404751  | **            | final_validation_loss |
| SGD           | ADEMAMIX      | 1.20353  | 0.682571 | ADEMAMIX | 1.00712e-07 | ***           | final_validation_loss |
| ADAMW         | ADAM_MINI     | 0.649415 | 1.78655  | ADAMW    | 1.20538e-08 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.649415 | 1.20712  | ADAMW    | 2.29447e-08 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.649415 | 1.46047  | ADAMW    | 3.47524e-05 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.649415 | 0.682571 | ADAMW    | 0.0375964   | *             | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      | 1.78655  | 1.20712  | NOVOGRAD | 8.17586e-08 | ***           | final_validation_loss |
| ADAM_MINI     | ADAGRAD       | 1.78655  | 1.46047  | ADAGRAD  | 0.000917921 | ***           | final_validation_loss |
| ADAM_MINI     | ADEMAMIX      | 1.78655  | 0.682571 | ADEMAMIX | 2.18417e-07 | ***           | final_validation_loss |
| NOVOGRAD      | ADAGRAD       | 1.20712  | 1.46047  | NOVOGRAD | 0.00385423  | **            | final_validation_loss |
| NOVOGRAD      | ADEMAMIX      | 1.20712  | 0.682571 | ADEMAMIX | 6.79852e-07 | ***           | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 1.46047  | 0.682571 | ADEMAMIX | 6.337e-05   | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 76.6163 | 0.7246 | 0.3241 | 75.7166 | 77.5160 |
| MILO_LW | 76.9956 | 0.7403 | 0.3311 | 76.0764 | 77.9147 |
| SGD | 73.5822 | 0.3282 | 0.1468 | 73.1747 | 73.9897 |
| ADAMW | 78.9956 | 1.0283 | 0.4599 | 77.7187 | 80.2724 |
| ADAM_MINI | 30.0741 | 2.5794 | 1.1535 | 26.8714 | 33.2768 |
| NOVOGRAD | 67.3630 | 0.8494 | 0.3798 | 66.3083 | 68.4176 |
| ADAGRAD | 74.0622 | 0.8882 | 0.3972 | 72.9593 | 75.1651 |
| ADEMAMIX | 77.5141 | 0.8965 | 0.4009 | 76.4010 | 78.6272 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  76.6163 |  76.9956 | MILO_LW  | 0.436685    |               | final_validation_accuracy |
| MILO          | SGD           |  76.6163 |  73.5822 | MILO     | 0.000210053 | ***           | final_validation_accuracy |
| MILO          | ADAMW         |  76.6163 |  78.9956 | ADAMW    | 0.00366823  | **            | final_validation_accuracy |
| MILO          | ADAM_MINI     |  76.6163 |  30.0741 | MILO     | 5.34876e-07 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  76.6163 |  67.363  | MILO     | 9.84197e-08 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  76.6163 |  74.0622 | MILO     | 0.00121067  | **            | final_validation_accuracy |
| MILO          | ADEMAMIX      |  76.6163 |  77.5141 | ADEMAMIX | 0.121416    |               | final_validation_accuracy |
| MILO_LW       | SGD           |  76.9956 |  73.5822 | MILO_LW  | 0.000132117 | ***           | final_validation_accuracy |
| MILO_LW       | ADAMW         |  76.9956 |  78.9956 | ADAMW    | 0.00903408  | **            | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  76.9956 |  30.0741 | MILO_LW  | 4.85141e-07 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  76.9956 |  67.363  | MILO_LW  | 7.22951e-08 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  76.9956 |  74.0622 | MILO_LW  | 0.000526389 | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  76.9956 |  77.5141 | ADEMAMIX | 0.34884     |               | final_validation_accuracy |
| SGD           | ADAMW         |  73.5822 |  78.9956 | ADAMW    | 0.000125376 | ***           | final_validation_accuracy |
| SGD           | ADAM_MINI     |  73.5822 |  30.0741 | SGD      | 2.19531e-06 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  73.5822 |  67.363  | SGD      | 1.68976e-05 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  73.5822 |  74.0622 | ADAGRAD  | 0.30773     |               | final_validation_accuracy |
| SGD           | ADEMAMIX      |  73.5822 |  77.5141 | ADEMAMIX | 0.000239568 | ***           | final_validation_accuracy |
| ADAMW         | ADAM_MINI     |  78.9956 |  30.0741 | ADAMW    | 1.10315e-07 | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  78.9956 |  67.363  | ADAMW    | 7.54691e-08 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  78.9956 |  74.0622 | ADAMW    | 4.42869e-05 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  78.9956 |  77.5141 | ADAMW    | 0.0418612   | *             | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  30.0741 |  67.363  | NOVOGRAD | 9.44182e-07 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  30.0741 |  74.0622 | ADAGRAD  | 3.60103e-07 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADEMAMIX      |  30.0741 |  77.5141 | ADEMAMIX | 2.39351e-07 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADAGRAD       |  67.363  |  74.0622 | ADAGRAD  | 1.93576e-06 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADEMAMIX      |  67.363  |  77.5141 | ADEMAMIX | 8.17428e-08 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  74.0622 |  77.5141 | ADEMAMIX | 0.000284458 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.7660 | 0.0070 | 0.0031 | 0.7573 | 0.7747 |
| MILO_LW | 0.7697 | 0.0069 | 0.0031 | 0.7611 | 0.7782 |
| SGD | 0.7320 | 0.0061 | 0.0027 | 0.7244 | 0.7395 |
| ADAMW | 0.7888 | 0.0112 | 0.0050 | 0.7749 | 0.8026 |
| ADAM_MINI | 0.2382 | 0.0397 | 0.0178 | 0.1889 | 0.2876 |
| NOVOGRAD | 0.6722 | 0.0099 | 0.0044 | 0.6599 | 0.6845 |
| ADAGRAD | 0.7416 | 0.0079 | 0.0035 | 0.7318 | 0.7515 |
| ADEMAMIX | 0.7776 | 0.0070 | 0.0031 | 0.7689 | 0.7863 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.766001 | 0.769654 | MILO_LW  | 0.428769    |               | final_validation_f1_score |
| MILO          | SGD           | 0.766001 | 0.731974 | MILO     | 3.98365e-05 | ***           | final_validation_f1_score |
| MILO          | ADAMW         | 0.766001 | 0.788791 | ADAMW    | 0.00663201  | **            | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.766001 | 0.238206 | MILO     | 4.64154e-06 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.766001 | 0.672208 | MILO     | 3.97429e-07 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.766001 | 0.741649 | MILO     | 0.000917067 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.766001 | 0.777591 | ADEMAMIX | 0.0307901   | *             | final_validation_f1_score |
| MILO_LW       | SGD           | 0.769654 | 0.731974 | MILO_LW  | 1.76235e-05 | ***           | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.769654 | 0.788791 | ADAMW    | 0.0147927   | *             | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.769654 | 0.238206 | MILO_LW  | 4.5725e-06  | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.769654 | 0.672208 | MILO_LW  | 3.1935e-07  | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.769654 | 0.741649 | MILO_LW  | 0.000364415 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.769654 | 0.777591 | ADEMAMIX | 0.10869     |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.731974 | 0.788791 | ADAMW    | 4.83362e-05 | ***           | final_validation_f1_score |
| SGD           | ADAM_MINI     | 0.731974 | 0.238206 | SGD      | 6.91611e-06 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.731974 | 0.672208 | SGD      | 1.25059e-05 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.731974 | 0.741649 | ADAGRAD  | 0.0645255   |               | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.731974 | 0.777591 | ADEMAMIX | 4.91971e-06 | ***           | final_validation_f1_score |
| ADAMW         | ADAM_MINI     | 0.788791 | 0.238206 | ADAMW    | 1.81279e-06 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.788791 | 0.672208 | ADAMW    | 1.37262e-07 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.788791 | 0.741649 | ADAMW    | 9.90285e-05 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.788791 | 0.777591 | ADAMW    | 0.100939    |               | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.238206 | 0.672208 | NOVOGRAD | 6.82615e-06 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.238206 | 0.741649 | ADAGRAD  | 4.92517e-06 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADEMAMIX      | 0.238206 | 0.777591 | ADEMAMIX | 4.20597e-06 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADAGRAD       | 0.672208 | 0.741649 | ADAGRAD  | 2.71635e-06 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADEMAMIX      | 0.672208 | 0.777591 | ADEMAMIX | 1.70489e-07 | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.741649 | 0.777591 | ADEMAMIX | 6.92376e-05 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9667 | 0.0011 | 0.0005 | 0.9654 | 0.9681 |
| MILO_LW | 0.9674 | 0.0018 | 0.0008 | 0.9651 | 0.9696 |
| SGD | 0.9626 | 0.0010 | 0.0005 | 0.9613 | 0.9639 |
| ADAMW | 0.9750 | 0.0011 | 0.0005 | 0.9736 | 0.9764 |
| ADAM_MINI | 0.8103 | 0.0120 | 0.0054 | 0.7955 | 0.8252 |
| NOVOGRAD | 0.9475 | 0.0012 | 0.0005 | 0.9460 | 0.9489 |
| ADAGRAD | 0.9616 | 0.0010 | 0.0005 | 0.9603 | 0.9628 |
| ADEMAMIX | 0.9714 | 0.0009 | 0.0004 | 0.9703 | 0.9725 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.966734 | 0.967374 | MILO_LW  | 0.520049    |               | final_validation_auc |
| MILO          | SGD           | 0.966734 | 0.962599 | MILO     | 0.000252951 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.966734 | 0.974985 | ADAMW    | 2.21003e-06 | ***           | final_validation_auc |
| MILO          | ADAM_MINI     | 0.966734 | 0.810328 | MILO     | 7.19556e-06 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.966734 | 0.947469 | MILO     | 4.61949e-09 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.966734 | 0.96158  | MILO     | 5.21196e-05 | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.966734 | 0.971431 | ADEMAMIX | 7.77701e-05 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.967374 | 0.962599 | MILO_LW  | 0.00184106  | **            | final_validation_auc |
| MILO_LW       | ADAMW         | 0.967374 | 0.974985 | ADAMW    | 0.000120621 | ***           | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.967374 | 0.810328 | MILO_LW  | 5.56605e-06 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.967374 | 0.947469 | MILO_LW  | 1.87925e-07 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.967374 | 0.96158  | MILO_LW  | 0.000657371 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.967374 | 0.971431 | ADEMAMIX | 0.00447451  | **            | final_validation_auc |
| SGD           | ADAMW         | 0.962599 | 0.974985 | ADAMW    | 8.67473e-08 | ***           | final_validation_auc |
| SGD           | ADAM_MINI     | 0.962599 | 0.810328 | SGD      | 8.10083e-06 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.962599 | 0.947469 | SGD      | 3.01255e-08 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.962599 | 0.96158  | SGD      | 0.152568    |               | final_validation_auc |
| SGD           | ADEMAMIX      | 0.962599 | 0.971431 | ADEMAMIX | 5.98379e-07 | ***           | final_validation_auc |
| ADAMW         | ADAM_MINI     | 0.974985 | 0.810328 | ADAMW    | 5.78754e-06 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.974985 | 0.947469 | ADAMW    | 2.86743e-10 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.974985 | 0.96158  | ADAMW    | 4.56487e-08 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.974985 | 0.971431 | ADAMW    | 0.000597451 | ***           | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.810328 | 0.947469 | NOVOGRAD | 1.1899e-05  | ***           | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.810328 | 0.96158  | ADAGRAD  | 8.35453e-06 | ***           | final_validation_auc |
| ADAM_MINI     | ADEMAMIX      | 0.810328 | 0.971431 | ADEMAMIX | 6.66935e-06 | ***           | final_validation_auc |
| NOVOGRAD      | ADAGRAD       | 0.947469 | 0.96158  | ADAGRAD  | 5.1832e-08  | ***           | final_validation_auc |
| NOVOGRAD      | ADEMAMIX      | 0.947469 | 0.971431 | ADEMAMIX | 1.49491e-09 | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.96158  | 0.971431 | ADEMAMIX | 2.32183e-07 | ***           | final_validation_auc |

