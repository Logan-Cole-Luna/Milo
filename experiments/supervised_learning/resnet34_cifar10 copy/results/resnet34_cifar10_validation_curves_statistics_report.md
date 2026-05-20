# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9292 | 0.0187 | 0.0083 | 0.9060 | 0.9523 |
| MILO_LW | 0.8555 | 0.0193 | 0.0086 | 0.8316 | 0.8794 |
| SGD | 1.1754 | 0.0634 | 0.0283 | 1.0968 | 1.2541 |
| ADAMW | 0.6574 | 0.0577 | 0.0258 | 0.5857 | 0.7290 |
| ADAM_MINI | 1.5586 | 0.0290 | 0.0130 | 1.5225 | 1.5946 |
| NOVOGRAD | 1.2277 | 0.0299 | 0.0134 | 1.1905 | 1.2649 |
| ADAGRAD | 1.0256 | 0.0134 | 0.0060 | 1.0090 | 1.0422 |
| ADEMAMIX | 0.7277 | 0.0609 | 0.0272 | 0.6521 | 0.8033 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.929153 | 0.855518 | MILO_LW  | 0.000277937 | ***           | final_validation_loss |
| MILO          | SGD           | 0.929153 | 1.17541  | MILO     | 0.000551519 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.929153 | 0.657355 | ADAMW    | 0.00020634  | ***           | final_validation_loss |
| MILO          | ADAM_MINI     | 0.929153 | 1.55858  | MILO     | 2.10527e-09 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      | 0.929153 | 1.22768  | MILO     | 4.59398e-07 | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.929153 | 1.02559  | MILO     | 2.57125e-05 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      | 0.929153 | 0.727687 | ADEMAMIX | 0.00108295  | **            | final_validation_loss |
| MILO_LW       | SGD           | 0.855518 | 1.17541  | MILO_LW  | 0.000163143 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         | 0.855518 | 0.657355 | ADAMW    | 0.0008459   | ***           | final_validation_loss |
| MILO_LW       | ADAM_MINI     | 0.855518 | 1.55858  | MILO_LW  | 7.76572e-10 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.855518 | 1.22768  | MILO_LW  | 9.05103e-08 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.855518 | 1.02559  | MILO_LW  | 6.90543e-07 | ***           | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.855518 | 0.727687 | ADEMAMIX | 0.00723933  | **            | final_validation_loss |
| SGD           | ADAMW         | 1.17541  | 0.657355 | ADAMW    | 9.32558e-07 | ***           | final_validation_loss |
| SGD           | ADAM_MINI     | 1.17541  | 1.55858  | SGD      | 2.86592e-05 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 1.17541  | 1.22768  | SGD      | 0.148922    |               | final_validation_loss |
| SGD           | ADAGRAD       | 1.17541  | 1.02559  | ADAGRAD  | 0.00524618  | **            | final_validation_loss |
| SGD           | ADEMAMIX      | 1.17541  | 0.727687 | ADEMAMIX | 3.22047e-06 | ***           | final_validation_loss |
| ADAMW         | ADAM_MINI     | 0.657355 | 1.55858  | ADAMW    | 8.85581e-08 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.657355 | 1.22768  | ADAMW    | 1.12218e-06 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.657355 | 1.02559  | ADAMW    | 8.04989e-05 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.657355 | 0.727687 | ADAMW    | 0.0978241   |               | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      | 1.55858  | 1.22768  | NOVOGRAD | 1.05528e-07 | ***           | final_validation_loss |
| ADAM_MINI     | ADAGRAD       | 1.55858  | 1.02559  | ADAGRAD  | 5.99117e-08 | ***           | final_validation_loss |
| ADAM_MINI     | ADEMAMIX      | 1.55858  | 0.727687 | ADEMAMIX | 2.61237e-07 | ***           | final_validation_loss |
| NOVOGRAD      | ADAGRAD       | 1.22768  | 1.02559  | ADAGRAD  | 1.71192e-05 | ***           | final_validation_loss |
| NOVOGRAD      | ADEMAMIX      | 1.22768  | 0.727687 | ADEMAMIX | 4.13243e-06 | ***           | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 1.02559  | 0.727687 | ADEMAMIX | 0.000264729 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 80.7763 | 0.8291 | 0.3708 | 79.7468 | 81.8058 |
| MILO_LW | 81.6741 | 0.5546 | 0.2480 | 80.9854 | 82.3627 |
| SGD | 70.1185 | 1.2150 | 0.5434 | 68.6099 | 71.6271 |
| ADAMW | 82.2904 | 0.8123 | 0.3633 | 81.2818 | 83.2990 |
| ADAM_MINI | 42.4948 | 0.9438 | 0.4221 | 41.3230 | 43.6667 |
| NOVOGRAD | 70.9689 | 0.6605 | 0.2954 | 70.1488 | 71.7890 |
| ADAGRAD | 64.1393 | 0.7480 | 0.3345 | 63.2105 | 65.0680 |
| ADEMAMIX | 81.6681 | 0.6594 | 0.2949 | 80.8495 | 82.4868 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  80.7763 |  81.6741 | MILO_LW  | 0.0841632   |               | final_validation_accuracy |
| MILO          | SGD           |  80.7763 |  70.1185 | MILO     | 7.61816e-07 | ***           | final_validation_accuracy |
| MILO          | ADAMW         |  80.7763 |  82.2904 | ADAMW    | 0.0193959   | *             | final_validation_accuracy |
| MILO          | ADAM_MINI     |  80.7763 |  42.4948 | MILO     | 3.42801e-12 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  80.7763 |  70.9689 | MILO     | 5.69714e-08 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  80.7763 |  64.1393 | MILO     | 8.52321e-10 | ***           | final_validation_accuracy |
| MILO          | ADEMAMIX      |  80.7763 |  81.6681 | ADEMAMIX | 0.0983988   |               | final_validation_accuracy |
| MILO_LW       | SGD           |  81.6741 |  70.1185 | MILO_LW  | 2.42438e-06 | ***           | final_validation_accuracy |
| MILO_LW       | ADAMW         |  81.6741 |  82.2904 | ADAMW    | 0.203557    |               | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  81.6741 |  42.4948 | MILO_LW  | 6.1643e-11  | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  81.6741 |  70.9689 | MILO_LW  | 4.72276e-09 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  81.6741 |  64.1393 | MILO_LW  | 4.6283e-10  | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  81.6741 |  81.6681 | MILO_LW  | 0.988117    |               | final_validation_accuracy |
| SGD           | ADAMW         |  70.1185 |  82.2904 | ADAMW    | 3.29335e-07 | ***           | final_validation_accuracy |
| SGD           | ADAM_MINI     |  70.1185 |  42.4948 | SGD      | 4.56364e-10 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  70.1185 |  70.9689 | NOVOGRAD | 0.21695     |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  70.1185 |  64.1393 | SGD      | 4.50853e-05 | ***           | final_validation_accuracy |
| SGD           | ADEMAMIX      |  70.1185 |  81.6681 | ADEMAMIX | 1.1572e-06  | ***           | final_validation_accuracy |
| ADAMW         | ADAM_MINI     |  82.2904 |  42.4948 | ADAMW    | 2.65976e-12 | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  82.2904 |  70.9689 | ADAMW    | 1.58495e-08 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  82.2904 |  64.1393 | ADAMW    | 3.69086e-10 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  82.2904 |  81.6681 | ADAMW    | 0.22172     |               | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  42.4948 |  70.9689 | NOVOGRAD | 1.09669e-10 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  42.4948 |  64.1393 | ADAGRAD  | 3.91576e-10 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADEMAMIX      |  42.4948 |  81.6681 | ADEMAMIX | 1.13887e-11 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADAGRAD       |  70.9689 |  64.1393 | NOVOGRAD | 3.85053e-07 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADEMAMIX      |  70.9689 |  81.6681 | ADEMAMIX | 5.75077e-09 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  64.1393 |  81.6681 | ADEMAMIX | 2.53345e-10 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.8072 | 0.0080 | 0.0036 | 0.7973 | 0.8171 |
| MILO_LW | 0.8161 | 0.0058 | 0.0026 | 0.8089 | 0.8234 |
| SGD | 0.6990 | 0.0115 | 0.0051 | 0.6848 | 0.7133 |
| ADAMW | 0.8231 | 0.0083 | 0.0037 | 0.8128 | 0.8335 |
| ADAM_MINI | 0.4126 | 0.0137 | 0.0061 | 0.3956 | 0.4296 |
| NOVOGRAD | 0.7075 | 0.0066 | 0.0029 | 0.6994 | 0.7156 |
| ADAGRAD | 0.6399 | 0.0081 | 0.0036 | 0.6298 | 0.6500 |
| ADEMAMIX | 0.8160 | 0.0068 | 0.0030 | 0.8075 | 0.8244 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.807175 | 0.816141 | MILO_LW  | 0.0796625   |               | final_validation_f1_score |
| MILO          | SGD           | 0.807175 | 0.699008 | MILO     | 4.39728e-07 | ***           | final_validation_f1_score |
| MILO          | ADAMW         | 0.807175 | 0.823149 | ADAMW    | 0.0146189   | *             | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.807175 | 0.412602 | MILO     | 7.33093e-10 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.807175 | 0.707506 | MILO     | 3.47966e-08 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.807175 | 0.639941 | MILO     | 8.10066e-10 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.807175 | 0.815972 | ADEMAMIX | 0.0978799   |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.816141 | 0.699008 | MILO_LW  | 1.02731e-06 | ***           | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.816141 | 0.823149 | ADAMW    | 0.165725    |               | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.816141 | 0.412602 | MILO_LW  | 7.38589e-09 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.816141 | 0.707506 | MILO_LW  | 3.81302e-09 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.816141 | 0.639941 | MILO_LW  | 1.0209e-09  | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.816141 | 0.815972 | MILO_LW  | 0.967246    |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.699008 | 0.823149 | ADAMW    | 1.42552e-07 | ***           | final_validation_f1_score |
| SGD           | ADAM_MINI     | 0.699008 | 0.412602 | SGD      | 6.70645e-10 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.699008 | 0.707506 | NOVOGRAD | 0.19788     |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.699008 | 0.639941 | SGD      | 2.68329e-05 | ***           | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.699008 | 0.815972 | ADEMAMIX | 5.00355e-07 | ***           | final_validation_f1_score |
| ADAMW         | ADAM_MINI     | 0.823149 | 0.412602 | ADAMW    | 3.83961e-10 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.823149 | 0.707506 | ADAMW    | 1.73576e-08 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.823149 | 0.639941 | ADAMW    | 4.67678e-10 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.823149 | 0.815972 | ADAMW    | 0.175034    |               | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.412602 | 0.707506 | NOVOGRAD | 1.9097e-08  | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.412602 | 0.639941 | ADAGRAD  | 2.14565e-08 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADEMAMIX      | 0.412602 | 0.815972 | ADEMAMIX | 2.38344e-09 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADAGRAD       | 0.707506 | 0.639941 | NOVOGRAD | 7.8897e-07  | ***           | final_validation_f1_score |
| NOVOGRAD      | ADEMAMIX      | 0.707506 | 0.815972 | ADEMAMIX | 5.78906e-09 | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.639941 | 0.815972 | ADEMAMIX | 5.16563e-10 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9770 | 0.0012 | 0.0005 | 0.9756 | 0.9785 |
| MILO_LW | 0.9792 | 0.0007 | 0.0003 | 0.9783 | 0.9802 |
| SGD | 0.9550 | 0.0034 | 0.0015 | 0.9508 | 0.9591 |
| ADAMW | 0.9833 | 0.0010 | 0.0005 | 0.9821 | 0.9846 |
| ADAM_MINI | 0.8572 | 0.0039 | 0.0017 | 0.8524 | 0.8620 |
| NOVOGRAD | 0.9562 | 0.0022 | 0.0010 | 0.9534 | 0.9589 |
| ADAGRAD | 0.9388 | 0.0022 | 0.0010 | 0.9361 | 0.9415 |
| ADEMAMIX | 0.9821 | 0.0012 | 0.0005 | 0.9806 | 0.9835 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.977042 | 0.979226 | MILO_LW  | 0.0101457   | *             | final_validation_auc |
| MILO          | SGD           | 0.977042 | 0.954965 | MILO     | 3.69733e-05 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.977042 | 0.983338 | ADAMW    | 1.94017e-05 | ***           | final_validation_auc |
| MILO          | ADAM_MINI     | 0.977042 | 0.857202 | MILO     | 3.37441e-08 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.977042 | 0.956191 | MILO     | 1.41361e-06 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.977042 | 0.938807 | MILO     | 2.5067e-08  | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.977042 | 0.982079 | ADEMAMIX | 0.000137193 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.979226 | 0.954965 | MILO_LW  | 4.87757e-05 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.979226 | 0.983338 | ADAMW    | 0.000130255 | ***           | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.979226 | 0.857202 | MILO_LW  | 1.0219e-07  | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.979226 | 0.956191 | MILO_LW  | 4.37735e-06 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.979226 | 0.938807 | MILO_LW  | 2.15756e-07 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.979226 | 0.982079 | ADEMAMIX | 0.00273275  | **            | final_validation_auc |
| SGD           | ADAMW         | 0.954965 | 0.983338 | ADAMW    | 1.50434e-05 | ***           | final_validation_auc |
| SGD           | ADAM_MINI     | 0.954965 | 0.857202 | SGD      | 1.42389e-10 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.954965 | 0.956191 | NOVOGRAD | 0.517456    |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.954965 | 0.938807 | SGD      | 4.79319e-05 | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.954965 | 0.982079 | ADEMAMIX | 1.34369e-05 | ***           | final_validation_auc |
| ADAMW         | ADAM_MINI     | 0.983338 | 0.857202 | ADAMW    | 4.30578e-08 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.983338 | 0.956191 | ADAMW    | 5.85923e-07 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.983338 | 0.938807 | ADAMW    | 2.55622e-08 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.983338 | 0.982079 | ADAMW    | 0.108082    |               | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.857202 | 0.956191 | NOVOGRAD | 1.70821e-09 | ***           | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.857202 | 0.938807 | ADAGRAD  | 7.32727e-09 | ***           | final_validation_auc |
| ADAM_MINI     | ADEMAMIX      | 0.857202 | 0.982079 | ADEMAMIX | 2.72752e-08 | ***           | final_validation_auc |
| NOVOGRAD      | ADAGRAD       | 0.956191 | 0.938807 | NOVOGRAD | 1.50388e-06 | ***           | final_validation_auc |
| NOVOGRAD      | ADEMAMIX      | 0.956191 | 0.982079 | ADEMAMIX | 3.7796e-07  | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.938807 | 0.982079 | ADEMAMIX | 1.13313e-08 | ***           | final_validation_auc |

