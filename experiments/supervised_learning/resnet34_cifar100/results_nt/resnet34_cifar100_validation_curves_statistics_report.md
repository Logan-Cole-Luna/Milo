# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 2.3846 | 0.0414 | 0.0185 | 2.3332 | 2.4360 |
| MILO_LW | 2.3963 | 0.0359 | 0.0161 | 2.3517 | 2.4409 |
| SGD | 2.5777 | 0.0575 | 0.0257 | 2.5063 | 2.6491 |
| ADAMW | 2.1527 | 0.0473 | 0.0212 | 2.0940 | 2.2114 |
| ADAGRAD | 2.3606 | 0.0882 | 0.0395 | 2.2510 | 2.4701 |
| ADEMAMIX | 2.2701 | 0.1174 | 0.0525 | 2.1243 | 2.4158 |
| SOAP | 1.9586 | 0.0378 | 0.0169 | 1.9117 | 2.0056 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  2.3846  |  2.3963  | MILO     | 0.646366    |               | final_validation_loss |
| MILO          | SGD           |  2.3846  |  2.57769 | MILO     | 0.000425834 | ***           | final_validation_loss |
| MILO          | ADAMW         |  2.3846  |  2.15269 | ADAMW    | 3.8636e-05  | ***           | final_validation_loss |
| MILO          | ADAGRAD       |  2.3846  |  2.36058 | ADAGRAD  | 0.602396    |               | final_validation_loss |
| MILO          | ADEMAMIX      |  2.3846  |  2.27006 | ADEMAMIX | 0.0948607   |               | final_validation_loss |
| MILO          | SOAP          |  2.3846  |  1.95862 | SOAP     | 1.59308e-07 | ***           | final_validation_loss |
| MILO_LW       | SGD           |  2.3963  |  2.57769 | MILO_LW  | 0.000646182 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         |  2.3963  |  2.15269 | ADAMW    | 2.52361e-05 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       |  2.3963  |  2.36058 | ADAGRAD  | 0.438028    |               | final_validation_loss |
| MILO_LW       | ADEMAMIX      |  2.3963  |  2.27006 | ADEMAMIX | 0.0726122   |               | final_validation_loss |
| MILO_LW       | SOAP          |  2.3963  |  1.95862 | SOAP     | 6.93049e-08 | ***           | final_validation_loss |
| SGD           | ADAMW         |  2.57769 |  2.15269 | ADAMW    | 1.84036e-06 | ***           | final_validation_loss |
| SGD           | ADAGRAD       |  2.57769 |  2.36058 | ADAGRAD  | 0.00256854  | **            | final_validation_loss |
| SGD           | ADEMAMIX      |  2.57769 |  2.27006 | ADEMAMIX | 0.0020858   | **            | final_validation_loss |
| SGD           | SOAP          |  2.57769 |  1.95862 | SOAP     | 2.15372e-07 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       |  2.15269 |  2.36058 | ADAMW    | 0.00334444  | **            | final_validation_loss |
| ADAMW         | ADEMAMIX      |  2.15269 |  2.27006 | ADAMW    | 0.0899015   |               | final_validation_loss |
| ADAMW         | SOAP          |  2.15269 |  1.95862 | SOAP     | 0.000120569 | ***           | final_validation_loss |
| ADAGRAD       | ADEMAMIX      |  2.36058 |  2.27006 | ADEMAMIX | 0.208097    |               | final_validation_loss |
| ADAGRAD       | SOAP          |  2.36058 |  1.95862 | SOAP     | 0.0001504   | ***           | final_validation_loss |
| ADEMAMIX      | SOAP          |  2.27006 |  1.95862 | SOAP     | 0.00271416  | **            | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 39.5141 | 1.1605 | 0.5190 | 38.0732 | 40.9550 |
| MILO_LW | 38.9926 | 0.6586 | 0.2946 | 38.1748 | 39.8104 |
| SGD | 34.0059 | 0.9193 | 0.4111 | 32.8644 | 35.1474 |
| ADAMW | 42.7111 | 1.0918 | 0.4883 | 41.3555 | 44.0667 |
| ADAGRAD | 39.5852 | 1.6193 | 0.7242 | 37.5746 | 41.5958 |
| ADEMAMIX | 40.3911 | 1.9633 | 0.8780 | 37.9533 | 42.8289 |
| SOAP | 54.4948 | 0.3636 | 0.1626 | 54.0434 | 54.9463 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  39.5141 |  38.9926 | MILO     | 0.414074    |               | final_validation_accuracy |
| MILO          | SGD           |  39.5141 |  34.0059 | MILO     | 4.43465e-05 | ***           | final_validation_accuracy |
| MILO          | ADAMW         |  39.5141 |  42.7111 | ADAMW    | 0.00205647  | **            | final_validation_accuracy |
| MILO          | ADAGRAD       |  39.5141 |  39.5852 | ADAGRAD  | 0.93854     |               | final_validation_accuracy |
| MILO          | ADEMAMIX      |  39.5141 |  40.3911 | ADEMAMIX | 0.420467    |               | final_validation_accuracy |
| MILO          | SOAP          |  39.5141 |  54.4948 | SOAP     | 1.89018e-06 | ***           | final_validation_accuracy |
| MILO_LW       | SGD           |  38.9926 |  34.0059 | MILO_LW  | 1.85776e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADAMW         |  38.9926 |  42.7111 | ADAMW    | 0.000427272 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  38.9926 |  39.5852 | ADAGRAD  | 0.480839    |               | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  38.9926 |  40.3911 | ADEMAMIX | 0.192706    |               | final_validation_accuracy |
| MILO_LW       | SOAP          |  38.9926 |  54.4948 | SOAP     | 3.93087e-09 | ***           | final_validation_accuracy |
| SGD           | ADAMW         |  34.0059 |  42.7111 | ADAMW    | 1.04872e-06 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  34.0059 |  39.5852 | ADAGRAD  | 0.000426583 | ***           | final_validation_accuracy |
| SGD           | ADEMAMIX      |  34.0059 |  40.3911 | ADEMAMIX | 0.000738024 | ***           | final_validation_accuracy |
| SGD           | SOAP          |  34.0059 |  54.4948 | SOAP     | 4.9569e-08  | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  42.7111 |  39.5852 | ADAMW    | 0.00895697  | **            | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  42.7111 |  40.3911 | ADAMW    | 0.0585664   |               | final_validation_accuracy |
| ADAMW         | SOAP          |  42.7111 |  54.4948 | SOAP     | 3.75149e-06 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  39.5852 |  40.3911 | ADEMAMIX | 0.49969     |               | final_validation_accuracy |
| ADAGRAD       | SOAP          |  39.5852 |  54.4948 | SOAP     | 1.68934e-05 | ***           | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  40.3911 |  54.4948 | SOAP     | 5.9395e-05  | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.3870 | 0.0142 | 0.0064 | 0.3693 | 0.4047 |
| MILO_LW | 0.3820 | 0.0069 | 0.0031 | 0.3734 | 0.3906 |
| SGD | 0.3302 | 0.0083 | 0.0037 | 0.3198 | 0.3405 |
| ADAMW | 0.4203 | 0.0109 | 0.0049 | 0.4068 | 0.4338 |
| ADAGRAD | 0.3930 | 0.0137 | 0.0061 | 0.3760 | 0.4100 |
| ADEMAMIX | 0.3954 | 0.0222 | 0.0099 | 0.3679 | 0.4229 |
| SOAP | 0.5438 | 0.0028 | 0.0012 | 0.5404 | 0.5473 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.387009 | 0.381978 | MILO     | 0.505207    |               | final_validation_f1_score |
| MILO          | SGD           | 0.387009 | 0.330171 | MILO     | 0.000175818 | ***           | final_validation_f1_score |
| MILO          | ADAMW         | 0.387009 | 0.420301 | ADAMW    | 0.00369508  | **            | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.387009 | 0.39304  | ADAGRAD  | 0.514173    |               | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.387009 | 0.395402 | ADEMAMIX | 0.500165    |               | final_validation_f1_score |
| MILO          | SOAP          | 0.387009 | 0.54385  | SOAP     | 9.1793e-06  | ***           | final_validation_f1_score |
| MILO_LW       | SGD           | 0.381978 | 0.330171 | MILO_LW  | 6.59621e-06 | ***           | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.381978 | 0.420301 | ADAMW    | 0.000331534 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.381978 | 0.39304  | ADAGRAD  | 0.158749    |               | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.381978 | 0.395402 | ADEMAMIX | 0.255507    |               | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.381978 | 0.54385  | SOAP     | 3.54352e-08 | ***           | final_validation_f1_score |
| SGD           | ADAMW         | 0.330171 | 0.420301 | ADAMW    | 8.44003e-07 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.330171 | 0.39304  | ADAGRAD  | 7.06776e-05 | ***           | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.330171 | 0.395402 | ADEMAMIX | 0.00152617  | **            | final_validation_f1_score |
| SGD           | SOAP          | 0.330171 | 0.54385  | SOAP     | 5.45461e-08 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.420301 | 0.39304  | ADAMW    | 0.00890744  | **            | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.420301 | 0.395402 | ADAMW    | 0.0664902   |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.420301 | 0.54385  | SOAP     | 5.3868e-06  | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.39304  | 0.395402 | ADEMAMIX | 0.8455      |               | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.39304  | 0.54385  | SOAP     | 8.75119e-06 | ***           | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.395402 | 0.54385  | SOAP     | 9.75865e-05 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9449 | 0.0022 | 0.0010 | 0.9422 | 0.9476 |
| MILO_LW | 0.9445 | 0.0018 | 0.0008 | 0.9423 | 0.9467 |
| SGD | 0.9379 | 0.0034 | 0.0015 | 0.9337 | 0.9422 |
| ADAMW | 0.9626 | 0.0025 | 0.0011 | 0.9596 | 0.9657 |
| ADAGRAD | 0.9496 | 0.0032 | 0.0014 | 0.9456 | 0.9535 |
| ADEMAMIX | 0.9565 | 0.0050 | 0.0023 | 0.9502 | 0.9627 |
| SOAP | 0.9718 | 0.0008 | 0.0003 | 0.9708 | 0.9727 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.944917 | 0.944477 | MILO     | 0.737926    |               | final_validation_auc |
| MILO          | SGD           | 0.944917 | 0.937945 | MILO     | 0.00644765  | **            | final_validation_auc |
| MILO          | ADAMW         | 0.944917 | 0.962639 | ADAMW    | 2.31492e-06 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.944917 | 0.949568 | ADAGRAD  | 0.0306163   | *             | final_validation_auc |
| MILO          | ADEMAMIX      | 0.944917 | 0.95645  | ADEMAMIX | 0.00427055  | **            | final_validation_auc |
| MILO          | SOAP          | 0.944917 | 0.97177  | SOAP     | 1.88757e-06 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.944477 | 0.937945 | MILO_LW  | 0.0086377   | **            | final_validation_auc |
| MILO_LW       | ADAMW         | 0.944477 | 0.962639 | ADAMW    | 2.08258e-06 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.944477 | 0.949568 | ADAGRAD  | 0.0192956   | *             | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.944477 | 0.95645  | ADEMAMIX | 0.00410225  | **            | final_validation_auc |
| MILO_LW       | SOAP          | 0.944477 | 0.97177  | SOAP     | 2.58301e-07 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.937945 | 0.962639 | ADAMW    | 2.35908e-06 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.937945 | 0.949568 | ADAGRAD  | 0.000520932 | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.937945 | 0.95645  | ADEMAMIX | 0.000249307 | ***           | final_validation_auc |
| SGD           | SOAP          | 0.937945 | 0.97177  | SOAP     | 1.19941e-05 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.962639 | 0.949568 | ADAMW    | 0.000117219 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.962639 | 0.95645  | ADAMW    | 0.0499016   | *             | final_validation_auc |
| ADAMW         | SOAP          | 0.962639 | 0.97177  | SOAP     | 0.000638309 | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.949568 | 0.95645  | ADEMAMIX | 0.0374849   | *             | final_validation_auc |
| ADAGRAD       | SOAP          | 0.949568 | 0.97177  | SOAP     | 5.29763e-05 | ***           | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.95645  | 0.97177  | SOAP     | 0.00217505  | **            | final_validation_auc |

