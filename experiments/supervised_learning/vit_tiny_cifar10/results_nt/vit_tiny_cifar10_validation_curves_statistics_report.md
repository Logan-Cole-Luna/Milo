# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 1.4273 | 0.0124 | 0.0055 | 1.4119 | 1.4426 |
| MILO_LW | 1.4483 | 0.0164 | 0.0073 | 1.4279 | 1.4686 |
| SGD | 1.5014 | 0.0108 | 0.0048 | 1.4880 | 1.5148 |
| ADAMW | 1.1578 | 0.0188 | 0.0084 | 1.1345 | 1.1811 |
| ADAGRAD | 1.3975 | 0.0256 | 0.0114 | 1.3657 | 1.4293 |
| ADEMAMIX | 1.1387 | 0.0088 | 0.0040 | 1.1277 | 1.1497 |
| SOAP | 0.8870 | 0.0150 | 0.0067 | 0.8684 | 0.9056 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  1.42725 |  1.44825 | MILO     | 0.0539175   |               | final_validation_loss |
| MILO          | SGD           |  1.42725 |  1.50138 | MILO     | 9.09198e-06 | ***           | final_validation_loss |
| MILO          | ADAMW         |  1.42725 |  1.15779 | ADAMW    | 2.93243e-08 | ***           | final_validation_loss |
| MILO          | ADAGRAD       |  1.42725 |  1.39751 | ADAGRAD  | 0.0595452   |               | final_validation_loss |
| MILO          | ADEMAMIX      |  1.42725 |  1.13872 | ADEMAMIX | 6.17405e-10 | ***           | final_validation_loss |
| MILO          | SOAP          |  1.42725 |  0.88702 | SOAP     | 1.03098e-11 | ***           | final_validation_loss |
| MILO_LW       | SGD           |  1.44825 |  1.50138 | MILO_LW  | 0.000535067 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         |  1.44825 |  1.15779 | ADAMW    | 6.48634e-09 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       |  1.44825 |  1.39751 | ADAGRAD  | 0.00770063  | **            | final_validation_loss |
| MILO_LW       | ADEMAMIX      |  1.44825 |  1.13872 | ADEMAMIX | 1.79977e-08 | ***           | final_validation_loss |
| MILO_LW       | SOAP          |  1.44825 |  0.88702 | SOAP     | 1.24756e-11 | ***           | final_validation_loss |
| SGD           | ADAMW         |  1.50138 |  1.15779 | ADAMW    | 1.41364e-08 | ***           | final_validation_loss |
| SGD           | ADAGRAD       |  1.50138 |  1.39751 | ADAGRAD  | 0.000279498 | ***           | final_validation_loss |
| SGD           | ADEMAMIX      |  1.50138 |  1.13872 | ADEMAMIX | 1.83058e-11 | ***           | final_validation_loss |
| SGD           | SOAP          |  1.50138 |  0.88702 | SOAP     | 9.60353e-12 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       |  1.15779 |  1.39751 | ADAMW    | 3.86226e-07 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      |  1.15779 |  1.13872 | ADEMAMIX | 0.0879603   |               | final_validation_loss |
| ADAMW         | SOAP          |  1.15779 |  0.88702 | SOAP     | 1.26672e-08 | ***           | final_validation_loss |
| ADAGRAD       | ADEMAMIX      |  1.39751 |  1.13872 | ADEMAMIX | 4.64631e-06 | ***           | final_validation_loss |
| ADAGRAD       | SOAP          |  1.39751 |  0.88702 | SOAP     | 7.19233e-09 | ***           | final_validation_loss |
| ADEMAMIX      | SOAP          |  1.13872 |  0.88702 | SOAP     | 2.06185e-08 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 48.8770 | 0.5457 | 0.2440 | 48.1995 | 49.5546 |
| MILO_LW | 48.1689 | 1.2137 | 0.5428 | 46.6619 | 49.6759 |
| SGD | 45.6059 | 0.7353 | 0.3288 | 44.6930 | 46.5189 |
| ADAMW | 58.2044 | 0.7804 | 0.3490 | 57.2354 | 59.1735 |
| ADAGRAD | 50.2015 | 0.8264 | 0.3696 | 49.1753 | 51.2276 |
| ADEMAMIX | 59.0222 | 0.6968 | 0.3116 | 58.1571 | 59.8874 |
| SOAP | 69.0637 | 0.7241 | 0.3238 | 68.1647 | 69.9627 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  48.877  |  48.1689 | MILO     | 0.282434    |               | final_validation_accuracy |
| MILO          | SGD           |  48.877  |  45.6059 | MILO     | 6.90596e-05 | ***           | final_validation_accuracy |
| MILO          | ADAMW         |  48.877  |  58.2044 | ADAMW    | 8.01966e-08 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  48.877  |  50.2015 | ADAGRAD  | 0.0204503   | *             | final_validation_accuracy |
| MILO          | ADEMAMIX      |  48.877  |  59.0222 | ADEMAMIX | 1.25201e-08 | ***           | final_validation_accuracy |
| MILO          | SOAP          |  48.877  |  69.0637 | SOAP     | 1.16925e-10 | ***           | final_validation_accuracy |
| MILO_LW       | SGD           |  48.1689 |  45.6059 | MILO_LW  | 0.00560793  | **            | final_validation_accuracy |
| MILO_LW       | ADAMW         |  48.1689 |  58.2044 | ADAMW    | 1.39615e-06 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  48.1689 |  50.2015 | ADAGRAD  | 0.0172698   | *             | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  48.1689 |  59.0222 | ADEMAMIX | 1.31831e-06 | ***           | final_validation_accuracy |
| MILO_LW       | SOAP          |  48.1689 |  69.0637 | SOAP     | 1.63176e-08 | ***           | final_validation_accuracy |
| SGD           | ADAMW         |  45.6059 |  58.2044 | ADAMW    | 4.98016e-09 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  45.6059 |  50.2015 | ADAGRAD  | 1.60371e-05 | ***           | final_validation_accuracy |
| SGD           | ADEMAMIX      |  45.6059 |  59.0222 | ADEMAMIX | 1.91456e-09 | ***           | final_validation_accuracy |
| SGD           | SOAP          |  45.6059 |  69.0637 | SOAP     | 2.49732e-11 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  58.2044 |  50.2015 | ADAMW    | 2.73892e-07 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  58.2044 |  59.0222 | ADEMAMIX | 0.119106    |               | final_validation_accuracy |
| ADAMW         | SOAP          |  58.2044 |  69.0637 | SOAP     | 1.55855e-08 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  50.2015 |  59.0222 | ADEMAMIX | 1.15553e-07 | ***           | final_validation_accuracy |
| ADAGRAD       | SOAP          |  50.2015 |  69.0637 | SOAP     | 3.1316e-10  | ***           | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  59.0222 |  69.0637 | SOAP     | 1.73525e-08 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.4847 | 0.0078 | 0.0035 | 0.4751 | 0.4943 |
| MILO_LW | 0.4745 | 0.0165 | 0.0074 | 0.4540 | 0.4950 |
| SGD | 0.4496 | 0.0088 | 0.0039 | 0.4387 | 0.4606 |
| ADAMW | 0.5747 | 0.0092 | 0.0041 | 0.5633 | 0.5862 |
| ADAGRAD | 0.5006 | 0.0081 | 0.0036 | 0.4906 | 0.5106 |
| ADEMAMIX | 0.5853 | 0.0083 | 0.0037 | 0.5750 | 0.5956 |
| SOAP | 0.6884 | 0.0080 | 0.0036 | 0.6785 | 0.6983 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.484715 | 0.474515 | MILO     | 0.259577    |               | final_validation_f1_score |
| MILO          | SGD           | 0.484715 | 0.449627 | MILO     | 0.000166043 | ***           | final_validation_f1_score |
| MILO          | ADAMW         | 0.484715 | 0.574724 | ADAMW    | 2.26859e-07 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.484715 | 0.500582 | ADAGRAD  | 0.0131647   | *             | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.484715 | 0.585277 | ADEMAMIX | 4.64839e-08 | ***           | final_validation_f1_score |
| MILO          | SOAP          | 0.484715 | 0.688409 | SOAP     | 1.40354e-10 | ***           | final_validation_f1_score |
| MILO_LW       | SGD           | 0.474515 | 0.449627 | MILO_LW  | 0.0241616   | *             | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.474515 | 0.574724 | ADAMW    | 1.56139e-05 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.474515 | 0.500582 | ADAGRAD  | 0.0200186   | *             | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.474515 | 0.585277 | ADEMAMIX | 1.20233e-05 | ***           | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.474515 | 0.688409 | SOAP     | 3.25001e-07 | ***           | final_validation_f1_score |
| SGD           | ADAMW         | 0.449627 | 0.574724 | ADAMW    | 2.01208e-08 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.449627 | 0.500582 | ADAGRAD  | 1.26214e-05 | ***           | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.449627 | 0.585277 | ADEMAMIX | 7.18059e-09 | ***           | final_validation_f1_score |
| SGD           | SOAP          | 0.449627 | 0.688409 | SOAP     | 7.91734e-11 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.574724 | 0.500582 | ADAMW    | 9.98858e-07 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.574724 | 0.585277 | ADEMAMIX | 0.0938866   |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.574724 | 0.688409 | SOAP     | 3.77641e-08 | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.500582 | 0.585277 | ADEMAMIX | 1.96278e-07 | ***           | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.500582 | 0.688409 | SOAP     | 3.08064e-10 | ***           | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.585277 | 0.688409 | SOAP     | 4.0662e-08  | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.8830 | 0.0021 | 0.0010 | 0.8803 | 0.8857 |
| MILO_LW | 0.8801 | 0.0024 | 0.0011 | 0.8772 | 0.8830 |
| SGD | 0.8685 | 0.0013 | 0.0006 | 0.8669 | 0.8701 |
| ADAMW | 0.9231 | 0.0013 | 0.0006 | 0.9214 | 0.9247 |
| ADAGRAD | 0.8902 | 0.0020 | 0.0009 | 0.8878 | 0.8927 |
| ADEMAMIX | 0.9249 | 0.0014 | 0.0006 | 0.9232 | 0.9267 |
| SOAP | 0.9543 | 0.0010 | 0.0004 | 0.9530 | 0.9555 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.882994 | 0.880113 | MILO     | 0.0780164   |               | final_validation_auc |
| MILO          | SGD           | 0.882994 | 0.868487 | MILO     | 6.65157e-06 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.882994 | 0.923076 | ADAMW    | 7.13113e-09 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.882994 | 0.890225 | ADAGRAD  | 0.000551965 | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.882994 | 0.924944 | ADEMAMIX | 3.61846e-09 | ***           | final_validation_auc |
| MILO          | SOAP          | 0.882994 | 0.954254 | SOAP     | 2.00919e-09 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.880113 | 0.868487 | MILO_LW  | 5.80833e-05 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.880113 | 0.923076 | ADAMW    | 1.5872e-08  | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.880113 | 0.890225 | ADAGRAD  | 9.20213e-05 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.880113 | 0.924944 | ADEMAMIX | 8.27084e-09 | ***           | final_validation_auc |
| MILO_LW       | SOAP          | 0.880113 | 0.954254 | SOAP     | 5.32047e-09 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.868487 | 0.923076 | ADAMW    | 3.21232e-12 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.868487 | 0.890225 | ADAGRAD  | 1.90001e-07 | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.868487 | 0.924944 | ADEMAMIX | 3.65058e-12 | ***           | final_validation_auc |
| SGD           | SOAP          | 0.868487 | 0.954254 | SOAP     | 1.29724e-13 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.923076 | 0.890225 | ADAMW    | 8.79748e-09 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.923076 | 0.924944 | ADEMAMIX | 0.0637941   |               | final_validation_auc |
| ADAMW         | SOAP          | 0.923076 | 0.954254 | SOAP     | 4.63219e-10 | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.890225 | 0.924944 | ADEMAMIX | 4.38809e-09 | ***           | final_validation_auc |
| ADAGRAD       | SOAP          | 0.890225 | 0.954254 | SOAP     | 1.11231e-09 | ***           | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.924944 | 0.954254 | SOAP     | 1.48451e-09 | ***           | final_validation_auc |

