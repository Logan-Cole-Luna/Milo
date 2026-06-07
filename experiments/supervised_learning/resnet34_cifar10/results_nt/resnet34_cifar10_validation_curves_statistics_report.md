# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.7706 | 0.0385 | 0.0172 | 0.7228 | 0.8183 |
| MILO_LW | 0.7629 | 0.0164 | 0.0073 | 0.7425 | 0.7832 |
| SGD | 1.0109 | 0.0907 | 0.0406 | 0.8982 | 1.1235 |
| ADAMW | 0.6723 | 0.0365 | 0.0163 | 0.6269 | 0.7176 |
| ADAGRAD | 0.9853 | 0.0651 | 0.0291 | 0.9045 | 1.0661 |
| ADEMAMIX | 0.6247 | 0.0314 | 0.0140 | 0.5858 | 0.6636 |
| SOAP | 0.5385 | 0.0124 | 0.0055 | 0.5231 | 0.5539 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.77058  | 0.762883 | MILO_LW  | 0.696344    |               | final_validation_loss |
| MILO          | SGD           | 0.77058  | 1.01086  | MILO     | 0.00222918  | **            | final_validation_loss |
| MILO          | ADAMW         | 0.77058  | 0.672265 | ADAMW    | 0.00325079  | **            | final_validation_loss |
| MILO          | ADAGRAD       | 0.77058  | 0.985278 | MILO     | 0.000521999 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      | 0.77058  | 0.624676 | ADEMAMIX | 0.000207089 | ***           | final_validation_loss |
| MILO          | SOAP          | 0.77058  | 0.538503 | SOAP     | 6.49922e-05 | ***           | final_validation_loss |
| MILO_LW       | SGD           | 0.762883 | 1.01086  | MILO_LW  | 0.00313632  | **            | final_validation_loss |
| MILO_LW       | ADAMW         | 0.762883 | 0.672265 | ADAMW    | 0.00290015  | **            | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.762883 | 0.985278 | MILO_LW  | 0.00109333  | **            | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.762883 | 0.624676 | ADEMAMIX | 0.000120571 | ***           | final_validation_loss |
| MILO_LW       | SOAP          | 0.762883 | 0.538503 | SOAP     | 2.21968e-08 | ***           | final_validation_loss |
| SGD           | ADAMW         | 1.01086  | 0.672265 | ADAMW    | 0.000454764 | ***           | final_validation_loss |
| SGD           | ADAGRAD       | 1.01086  | 0.985278 | ADAGRAD  | 0.623669    |               | final_validation_loss |
| SGD           | ADEMAMIX      | 1.01086  | 0.624676 | ADEMAMIX | 0.000301094 | ***           | final_validation_loss |
| SGD           | SOAP          | 1.01086  | 0.538503 | SOAP     | 0.000262732 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.672265 | 0.985278 | ADAMW    | 6.26334e-05 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.672265 | 0.624676 | ADEMAMIX | 0.0587998   |               | final_validation_loss |
| ADAMW         | SOAP          | 0.672265 | 0.538503 | SOAP     | 0.000620159 | ***           | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 0.985278 | 0.624676 | ADEMAMIX | 4.04132e-05 | ***           | final_validation_loss |
| ADAGRAD       | SOAP          | 0.985278 | 0.538503 | SOAP     | 7.0237e-05  | ***           | final_validation_loss |
| ADEMAMIX      | SOAP          | 0.624676 | 0.538503 | SOAP     | 0.00198738  | **            | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 75.4756 | 1.5369 | 0.6873 | 73.5672 | 77.3839 |
| MILO_LW | 75.4044 | 0.5327 | 0.2382 | 74.7431 | 76.0658 |
| SGD | 67.6267 | 2.5722 | 1.1503 | 64.4329 | 70.8205 |
| ADAMW | 77.3985 | 1.5234 | 0.6813 | 75.5069 | 79.2901 |
| ADAGRAD | 72.0622 | 1.2561 | 0.5617 | 70.5026 | 73.6219 |
| ADEMAMIX | 78.7763 | 1.4246 | 0.6371 | 77.0074 | 80.5452 |
| SOAP | 84.1956 | 0.4698 | 0.2101 | 83.6122 | 84.7789 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  75.4756 |  75.4044 | MILO     | 0.925963    |               | final_validation_accuracy |
| MILO          | SGD           |  75.4756 |  67.6267 | MILO     | 0.000805802 | ***           | final_validation_accuracy |
| MILO          | ADAMW         |  75.4756 |  77.3985 | ADAMW    | 0.0821569   |               | final_validation_accuracy |
| MILO          | ADAGRAD       |  75.4756 |  72.0622 | MILO     | 0.00528616  | **            | final_validation_accuracy |
| MILO          | ADEMAMIX      |  75.4756 |  78.7763 | ADEMAMIX | 0.00789549  | **            | final_validation_accuracy |
| MILO          | SOAP          |  75.4756 |  84.1956 | SOAP     | 9.47376e-05 | ***           | final_validation_accuracy |
| MILO_LW       | SGD           |  75.4044 |  67.6267 | MILO_LW  | 0.00200902  | **            | final_validation_accuracy |
| MILO_LW       | ADAMW         |  75.4044 |  77.3985 | ADAMW    | 0.0400138   | *             | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  75.4044 |  72.0622 | MILO_LW  | 0.0021789   | **            | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  75.4044 |  78.7763 | ADEMAMIX | 0.00403977  | **            | final_validation_accuracy |
| MILO_LW       | SOAP          |  75.4044 |  84.1956 | SOAP     | 3.93443e-09 | ***           | final_validation_accuracy |
| SGD           | ADAMW         |  67.6267 |  77.3985 | ADAMW    | 0.000230609 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  67.6267 |  72.0622 | ADAGRAD  | 0.0141146   | *             | final_validation_accuracy |
| SGD           | ADEMAMIX      |  67.6267 |  78.7763 | ADEMAMIX | 0.000118791 | ***           | final_validation_accuracy |
| SGD           | SOAP          |  67.6267 |  84.1956 | SOAP     | 9.48619e-05 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  77.3985 |  72.0622 | ADAMW    | 0.000354898 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  77.3985 |  78.7763 | ADEMAMIX | 0.178074    |               | final_validation_accuracy |
| ADAMW         | SOAP          |  77.3985 |  84.1956 | SOAP     | 0.000281728 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  72.0622 |  78.7763 | ADEMAMIX | 5.19053e-05 | ***           | final_validation_accuracy |
| ADAGRAD       | SOAP          |  72.0622 |  84.1956 | SOAP     | 4.57763e-06 | ***           | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  78.7763 |  84.1956 | SOAP     | 0.000537872 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.7546 | 0.0156 | 0.0070 | 0.7352 | 0.7740 |
| MILO_LW | 0.7532 | 0.0053 | 0.0024 | 0.7467 | 0.7598 |
| SGD | 0.6745 | 0.0293 | 0.0131 | 0.6381 | 0.7109 |
| ADAMW | 0.7733 | 0.0143 | 0.0064 | 0.7556 | 0.7910 |
| ADAGRAD | 0.7193 | 0.0160 | 0.0072 | 0.6994 | 0.7392 |
| ADEMAMIX | 0.7860 | 0.0136 | 0.0061 | 0.7691 | 0.8030 |
| SOAP | 0.8417 | 0.0048 | 0.0021 | 0.8358 | 0.8476 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.754604 | 0.753247 | MILO     | 0.861487    |               | final_validation_f1_score |
| MILO          | SGD           | 0.754604 | 0.67446  | MILO     | 0.00158523  | **            | final_validation_f1_score |
| MILO          | ADAMW         | 0.754604 | 0.77329  | ADAMW    | 0.0840129   |               | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.754604 | 0.719335 | MILO     | 0.00782384  | **            | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.754604 | 0.786037 | ADEMAMIX | 0.00978721  | **            | final_validation_f1_score |
| MILO          | SOAP          | 0.754604 | 0.841706 | SOAP     | 0.000103744 | ***           | final_validation_f1_score |
| MILO_LW       | SGD           | 0.753247 | 0.67446  | MILO_LW  | 0.00335626  | **            | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.753247 | 0.77329  | ADAMW    | 0.0313468   | *             | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.753247 | 0.719335 | MILO_LW  | 0.00689711  | **            | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.753247 | 0.786037 | ADEMAMIX | 0.0036899   | **            | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.753247 | 0.841706 | SOAP     | 3.42375e-09 | ***           | final_validation_f1_score |
| SGD           | ADAMW         | 0.67446  | 0.77329  | ADAMW    | 0.000585403 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.67446  | 0.719335 | ADAGRAD  | 0.0230074   | *             | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.67446  | 0.786037 | ADEMAMIX | 0.000331314 | ***           | final_validation_f1_score |
| SGD           | SOAP          | 0.67446  | 0.841706 | SOAP     | 0.000168584 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.77329  | 0.719335 | ADAMW    | 0.000519408 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.77329  | 0.786037 | ADEMAMIX | 0.186442    |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.77329  | 0.841706 | SOAP     | 0.000179759 | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.719335 | 0.786037 | ADEMAMIX | 0.000116547 | ***           | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.719335 | 0.841706 | SOAP     | 2.51505e-05 | ***           | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.786037 | 0.841706 | SOAP     | 0.000360172 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9683 | 0.0026 | 0.0012 | 0.9651 | 0.9716 |
| MILO_LW | 0.9683 | 0.0009 | 0.0004 | 0.9673 | 0.9694 |
| SGD | 0.9540 | 0.0028 | 0.0013 | 0.9505 | 0.9575 |
| ADAMW | 0.9744 | 0.0024 | 0.0011 | 0.9715 | 0.9774 |
| ADAGRAD | 0.9619 | 0.0019 | 0.0008 | 0.9595 | 0.9642 |
| ADEMAMIX | 0.9767 | 0.0021 | 0.0010 | 0.9740 | 0.9793 |
| SOAP | 0.9856 | 0.0004 | 0.0002 | 0.9851 | 0.9861 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.96835  | 0.968326 | MILO     | 0.985574    |               | final_validation_auc |
| MILO          | SGD           | 0.96835  | 0.95401  | MILO     | 3.51479e-05 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.96835  | 0.974432 | ADAMW    | 0.00520258  | **            | final_validation_auc |
| MILO          | ADAGRAD       | 0.96835  | 0.961858 | MILO     | 0.00264566  | **            | final_validation_auc |
| MILO          | ADEMAMIX      | 0.96835  | 0.976671 | ADEMAMIX | 0.000677048 | ***           | final_validation_auc |
| MILO          | SOAP          | 0.96835  | 0.985596 | SOAP     | 9.92506e-05 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.968326 | 0.95401  | MILO_LW  | 0.000165322 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.968326 | 0.974432 | ADAMW    | 0.0030097   | **            | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.968326 | 0.961858 | MILO_LW  | 0.000575149 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.968326 | 0.976671 | ADEMAMIX | 0.000361845 | ***           | final_validation_auc |
| MILO_LW       | SOAP          | 0.968326 | 0.985596 | SOAP     | 2.99141e-08 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.95401  | 0.974432 | ADAMW    | 2.24248e-06 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.95401  | 0.961858 | ADAGRAD  | 0.00134226  | **            | final_validation_auc |
| SGD           | ADEMAMIX      | 0.95401  | 0.976671 | ADEMAMIX | 1.13994e-06 | ***           | final_validation_auc |
| SGD           | SOAP          | 0.95401  | 0.985596 | SOAP     | 1.14058e-05 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.974432 | 0.961858 | ADAMW    | 2.18856e-05 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.974432 | 0.976671 | ADEMAMIX | 0.157293    |               | final_validation_auc |
| ADAMW         | SOAP          | 0.974432 | 0.985596 | SOAP     | 0.000379143 | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.961858 | 0.976671 | ADEMAMIX | 3.03816e-06 | ***           | final_validation_auc |
| ADAGRAD       | SOAP          | 0.961858 | 0.985596 | SOAP     | 4.54652e-06 | ***           | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.976671 | 0.985596 | SOAP     | 0.000557722 | ***           | final_validation_auc |

