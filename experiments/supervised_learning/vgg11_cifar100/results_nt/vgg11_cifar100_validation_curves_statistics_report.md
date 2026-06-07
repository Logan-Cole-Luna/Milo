# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 2.6654 | 0.0644 | 0.0288 | 2.5855 | 2.7453 |
| MILO_LW | 2.6286 | 0.0315 | 0.0141 | 2.5896 | 2.6677 |
| SGD | 2.7220 | 0.0236 | 0.0106 | 2.6927 | 2.7513 |
| ADAMW | 3.1630 | 0.0687 | 0.0307 | 3.0778 | 3.2483 |
| ADAGRAD | 2.5899 | 0.0617 | 0.0276 | 2.5133 | 2.6665 |
| ADEMAMIX | 3.3755 | 0.0462 | 0.0206 | 3.3182 | 3.4329 |
| SOAP | 173590913.4230 | 388161075.9857 | 173590910.4247 | -308374720.1326 | 655556546.9785 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |      Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|------------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       |  2.66542 | 2.62862     | MILO_LW  | 0.295782    |               | final_validation_loss |
| MILO          | SGD           |  2.66542 | 2.72201     | MILO     | 0.123565    |               | final_validation_loss |
| MILO          | ADAMW         |  2.66542 | 3.16304     | MILO     | 2.48314e-06 | ***           | final_validation_loss |
| MILO          | ADAGRAD       |  2.66542 | 2.58993     | ADAGRAD  | 0.0949777   |               | final_validation_loss |
| MILO          | ADEMAMIX      |  2.66542 | 3.37554     | MILO     | 1.27988e-07 | ***           | final_validation_loss |
| MILO          | SOAP          |  2.66542 | 1.73591e+08 | MILO     | 0.373901    |               | final_validation_loss |
| MILO_LW       | SGD           |  2.62862 | 2.72201     | MILO_LW  | 0.000922046 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         |  2.62862 | 3.16304     | MILO_LW  | 7.21352e-06 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       |  2.62862 | 2.58993     | ADAGRAD  | 0.258444    |               | final_validation_loss |
| MILO_LW       | ADEMAMIX      |  2.62862 | 3.37554     | MILO_LW  | 1.07853e-08 | ***           | final_validation_loss |
| MILO_LW       | SOAP          |  2.62862 | 1.73591e+08 | MILO_LW  | 0.373901    |               | final_validation_loss |
| SGD           | ADAMW         |  2.72201 | 3.16304     | SGD      | 4.26082e-05 | ***           | final_validation_loss |
| SGD           | ADAGRAD       |  2.72201 | 2.58993     | ADAGRAD  | 0.0061359   | **            | final_validation_loss |
| SGD           | ADEMAMIX      |  2.72201 | 3.37554     | SGD      | 1.4337e-07  | ***           | final_validation_loss |
| SGD           | SOAP          |  2.72201 | 1.73591e+08 | SGD      | 0.373901    |               | final_validation_loss |
| ADAMW         | ADAGRAD       |  3.16304 | 2.58993     | ADAGRAD  | 7.79821e-07 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      |  3.16304 | 3.37554     | ADAMW    | 0.000702135 | ***           | final_validation_loss |
| ADAMW         | SOAP          |  3.16304 | 1.73591e+08 | ADAMW    | 0.373901    |               | final_validation_loss |
| ADAGRAD       | ADEMAMIX      |  2.58993 | 3.37554     | ADAGRAD  | 3.9182e-08  | ***           | final_validation_loss |
| ADAGRAD       | SOAP          |  2.58993 | 1.73591e+08 | ADAGRAD  | 0.373901    |               | final_validation_loss |
| ADEMAMIX      | SOAP          |  3.37554 | 1.73591e+08 | ADEMAMIX | 0.373901    |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 31.7867 | 1.3828 | 0.6184 | 30.0697 | 33.5036 |
| MILO_LW | 32.5778 | 0.5814 | 0.2600 | 31.8559 | 33.2997 |
| SGD | 29.9941 | 0.6731 | 0.3010 | 29.1583 | 30.8298 |
| ADAMW | 17.9378 | 1.3774 | 0.6160 | 16.2275 | 19.6481 |
| ADAGRAD | 34.9452 | 0.7830 | 0.3502 | 33.9730 | 35.9174 |
| ADEMAMIX | 14.0593 | 0.6017 | 0.2691 | 13.3122 | 14.8064 |
| SOAP | 28.3259 | 24.9403 | 11.1536 | -2.6415 | 59.2934 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  31.7867 |  32.5778 | MILO_LW  | 0.287861    |               | final_validation_accuracy |
| MILO          | SGD           |  31.7867 |  29.9941 | MILO     | 0.0416335   | *             | final_validation_accuracy |
| MILO          | ADAMW         |  31.7867 |  17.9378 | MILO     | 2.49218e-07 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  31.7867 |  34.9452 | ADAGRAD  | 0.0038301   | **            | final_validation_accuracy |
| MILO          | ADEMAMIX      |  31.7867 |  14.0593 | MILO     | 5.81165e-07 | ***           | final_validation_accuracy |
| MILO          | SOAP          |  31.7867 |  28.3259 | MILO     | 0.772089    |               | final_validation_accuracy |
| MILO_LW       | SGD           |  32.5778 |  29.9941 | MILO_LW  | 0.000207056 | ***           | final_validation_accuracy |
| MILO_LW       | ADAMW         |  32.5778 |  17.9378 | MILO_LW  | 1.81488e-06 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  32.5778 |  34.9452 | ADAGRAD  | 0.000819248 | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  32.5778 |  14.0593 | MILO_LW  | 3.14605e-11 | ***           | final_validation_accuracy |
| MILO_LW       | SOAP          |  32.5778 |  28.3259 | MILO_LW  | 0.722483    |               | final_validation_accuracy |
| SGD           | ADAMW         |  29.9941 |  17.9378 | SGD      | 2.94014e-06 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  29.9941 |  34.9452 | ADAGRAD  | 5.95759e-06 | ***           | final_validation_accuracy |
| SGD           | ADEMAMIX      |  29.9941 |  14.0593 | SGD      | 2.32105e-10 | ***           | final_validation_accuracy |
| SGD           | SOAP          |  29.9941 |  28.3259 | SGD      | 0.888379    |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  17.9378 |  34.9452 | ADAGRAD  | 1.82794e-07 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  17.9378 |  14.0593 | ADAMW    | 0.00162346  | **            | final_validation_accuracy |
| ADAMW         | SOAP          |  17.9378 |  28.3259 | SOAP     | 0.404739    |               | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  34.9452 |  14.0593 | ADAGRAD  | 1.45386e-10 | ***           | final_validation_accuracy |
| ADAGRAD       | SOAP          |  34.9452 |  28.3259 | ADAGRAD  | 0.584921    |               | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  14.0593 |  28.3259 | SOAP     | 0.270067    |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.2985 | 0.0150 | 0.0067 | 0.2799 | 0.3171 |
| MILO_LW | 0.3088 | 0.0090 | 0.0040 | 0.2976 | 0.3200 |
| SGD | 0.2796 | 0.0106 | 0.0048 | 0.2664 | 0.2928 |
| ADAMW | 0.1361 | 0.0151 | 0.0067 | 0.1174 | 0.1548 |
| ADAGRAD | 0.3367 | 0.0076 | 0.0034 | 0.3273 | 0.3462 |
| ADEMAMIX | 0.0997 | 0.0073 | 0.0033 | 0.0905 | 0.1088 |
| SOAP | 0.2767 | 0.2520 | 0.1127 | -0.0362 | 0.5897 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |    Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|----------:|----------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.298513  | 0.308834  | MILO_LW  | 0.231265    |               | final_validation_f1_score |
| MILO          | SGD           | 0.298513  | 0.279634  | MILO     | 0.0541247   |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.298513  | 0.136098  | MILO     | 1.40671e-07 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.298513  | 0.336748  | ADAGRAD  | 0.00232164  | **            | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.298513  | 0.0996553 | MILO     | 2.65812e-07 | ***           | final_validation_f1_score |
| MILO          | SOAP          | 0.298513  | 0.276736  | MILO     | 0.856386    |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.308834  | 0.279634  | MILO_LW  | 0.00168306  | **            | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.308834  | 0.136098  | MILO_LW  | 2.27297e-07 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.308834  | 0.336748  | ADAGRAD  | 0.000812389 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.308834  | 0.0996553 | MILO_LW  | 3.23189e-10 | ***           | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.308834  | 0.276736  | MILO_LW  | 0.790033    |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.279634  | 0.136098  | SGD      | 3.90639e-07 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.279634  | 0.336748  | ADAGRAD  | 1.96223e-05 | ***           | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.279634  | 0.0996553 | SGD      | 7.18171e-09 | ***           | final_validation_f1_score |
| SGD           | SOAP          | 0.279634  | 0.276736  | SGD      | 0.980732    |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.136098  | 0.336748  | ADAGRAD  | 2.19101e-07 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.136098  | 0.0996553 | ADAMW    | 0.00311918  | **            | final_validation_f1_score |
| ADAMW         | SOAP          | 0.136098  | 0.276736  | SOAP     | 0.280433    |               | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.336748  | 0.0996553 | ADAGRAD  | 2.91994e-11 | ***           | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.336748  | 0.276736  | ADAGRAD  | 0.622709    |               | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.0996553 | 0.276736  | SOAP     | 0.191262    |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9310 | 0.0038 | 0.0017 | 0.9263 | 0.9358 |
| MILO_LW | 0.9333 | 0.0019 | 0.0008 | 0.9310 | 0.9356 |
| SGD | 0.9311 | 0.0022 | 0.0010 | 0.9284 | 0.9338 |
| ADAMW | 0.8984 | 0.0069 | 0.0031 | 0.8898 | 0.9070 |
| ADAGRAD | 0.9400 | 0.0021 | 0.0009 | 0.9374 | 0.9426 |
| ADEMAMIX | 0.8765 | 0.0049 | 0.0022 | 0.8704 | 0.8826 |
| SOAP | 0.7661 | 0.2420 | 0.1082 | 0.4655 | 1.0666 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.931041 | 0.933305 | MILO_LW  | 0.282203    |               | final_validation_auc |
| MILO          | SGD           | 0.931041 | 0.931132 | SGD      | 0.964906    |               | final_validation_auc |
| MILO          | ADAMW         | 0.931041 | 0.898359 | MILO     | 7.19698e-05 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.931041 | 0.939993 | ADAGRAD  | 0.00350755  | **            | final_validation_auc |
| MILO          | ADEMAMIX      | 0.931041 | 0.876489 | MILO     | 9.98962e-08 | ***           | final_validation_auc |
| MILO          | SOAP          | 0.931041 | 0.766065 | MILO     | 0.202181    |               | final_validation_auc |
| MILO_LW       | SGD           | 0.933305 | 0.931132 | MILO_LW  | 0.129137    |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.933305 | 0.898359 | MILO_LW  | 0.000191865 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.933305 | 0.939993 | ADAGRAD  | 0.000765164 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.933305 | 0.876489 | MILO_LW  | 1.86876e-06 | ***           | final_validation_auc |
| MILO_LW       | SOAP          | 0.933305 | 0.766065 | MILO_LW  | 0.197242    |               | final_validation_auc |
| SGD           | ADAMW         | 0.931132 | 0.898359 | SGD      | 0.000209121 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.931132 | 0.939993 | ADAGRAD  | 0.000186569 | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.931132 | 0.876489 | SGD      | 1.21244e-06 | ***           | final_validation_auc |
| SGD           | SOAP          | 0.931132 | 0.766065 | SGD      | 0.201974    |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.898359 | 0.939993 | ADAGRAD  | 7.22466e-05 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.898359 | 0.876489 | ADAMW    | 0.00062085  | ***           | final_validation_auc |
| ADAMW         | SOAP          | 0.898359 | 0.766065 | ADAMW    | 0.288812    |               | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.939993 | 0.876489 | ADAGRAD  | 6.08126e-07 | ***           | final_validation_auc |
| ADAGRAD       | SOAP          | 0.939993 | 0.766065 | ADAGRAD  | 0.183386    |               | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.876489 | 0.766065 | ADEMAMIX | 0.365401    |               | final_validation_auc |

