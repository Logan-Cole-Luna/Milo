# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.3214 | 0.0005 | 0.0002 | 0.3207 | 0.3220 |
| MILO_LW | 0.3206 | 0.0016 | 0.0007 | 0.3186 | 0.3227 |
| SGD | 0.3020 | 0.0014 | 0.0006 | 0.3002 | 0.3037 |
| ADAMW | 0.5219 | 0.0201 | 0.0090 | 0.4969 | 0.5469 |
| ADAGRAD | 0.2910 | 0.0009 | 0.0004 | 0.2899 | 0.2921 |
| ADEMAMIX | 0.5105 | 0.0549 | 0.0245 | 0.4424 | 0.5787 |
| SOAP | 0.3897 | 0.0029 | 0.0013 | 0.3860 | 0.3933 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.321354 | 0.320638 | MILO_LW  | 0.395031    |               | final_validation_loss |
| MILO          | SGD           | 0.321354 | 0.301979 | SGD      | 1.03985e-06 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.321354 | 0.521861 | MILO     | 2.38582e-05 | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.321354 | 0.291021 | ADAGRAD  | 5.03442e-10 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      | 0.321354 | 0.510536 | MILO     | 0.00152497  | **            | final_validation_loss |
| MILO          | SOAP          | 0.321354 | 0.389667 | MILO     | 4.3765e-07  | ***           | final_validation_loss |
| MILO_LW       | SGD           | 0.320638 | 0.301979 | SGD      | 7.00985e-08 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         | 0.320638 | 0.521861 | MILO_LW  | 2.1593e-05  | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.320638 | 0.291021 | ADAGRAD  | 1.98535e-08 | ***           | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.320638 | 0.510536 | MILO_LW  | 0.0014957   | **            | final_validation_loss |
| MILO_LW       | SOAP          | 0.320638 | 0.389667 | MILO_LW  | 3.29299e-09 | ***           | final_validation_loss |
| SGD           | ADAMW         | 0.301979 | 0.521861 | SGD      | 1.54831e-05 | ***           | final_validation_loss |
| SGD           | ADAGRAD       | 0.301979 | 0.291021 | ADAGRAD  | 2.25447e-06 | ***           | final_validation_loss |
| SGD           | ADEMAMIX      | 0.301979 | 0.510536 | SGD      | 0.00104773  | **            | final_validation_loss |
| SGD           | SOAP          | 0.301979 | 0.389667 | SGD      | 2.43289e-09 | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.521861 | 0.291021 | ADAGRAD  | 1.33207e-05 | ***           | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.521861 | 0.510536 | ADEMAMIX | 0.682786    |               | final_validation_loss |
| ADAMW         | SOAP          | 0.521861 | 0.389667 | SOAP     | 9.97444e-05 | ***           | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 0.291021 | 0.510536 | ADAGRAD  | 0.000863281 | ***           | final_validation_loss |
| ADAGRAD       | SOAP          | 0.291021 | 0.389667 | ADAGRAD  | 1.96969e-08 | ***           | final_validation_loss |
| ADEMAMIX      | SOAP          | 0.510536 | 0.389667 | SOAP     | 0.00783384  | **            | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 91.4198 | 0.0907 | 0.0406 | 91.3071 | 91.5324 |
| MILO_LW | 91.5136 | 0.0897 | 0.0401 | 91.4022 | 91.6250 |
| SGD | 91.6667 | 0.0720 | 0.0322 | 91.5773 | 91.7561 |
| ADAMW | 89.2099 | 0.3352 | 0.1499 | 88.7937 | 89.6260 |
| ADAGRAD | 92.0321 | 0.1539 | 0.0688 | 91.8411 | 92.2231 |
| ADEMAMIX | 88.1605 | 1.7531 | 0.7840 | 85.9837 | 90.3373 |
| SOAP | 90.1383 | 0.3060 | 0.1368 | 89.7584 | 90.5182 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  91.4198 |  91.5136 | MILO_LW  | 0.138715    |               | final_validation_accuracy |
| MILO          | SGD           |  91.4198 |  91.6667 | SGD      | 0.00162462  | **            | final_validation_accuracy |
| MILO          | ADAMW         |  91.4198 |  89.2099 | MILO     | 5.73616e-05 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  91.4198 |  92.0321 | ADAGRAD  | 0.000176337 | ***           | final_validation_accuracy |
| MILO          | ADEMAMIX      |  91.4198 |  88.1605 | MILO     | 0.0140857   | *             | final_validation_accuracy |
| MILO          | SOAP          |  91.4198 |  90.1383 | MILO     | 0.000392565 | ***           | final_validation_accuracy |
| MILO_LW       | SGD           |  91.5136 |  91.6667 | SGD      | 0.0186477   | *             | final_validation_accuracy |
| MILO_LW       | ADAMW         |  91.5136 |  89.2099 | MILO_LW  | 4.83633e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  91.5136 |  92.0321 | ADAGRAD  | 0.000469485 | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  91.5136 |  88.1605 | MILO_LW  | 0.0127934   | *             | final_validation_accuracy |
| MILO_LW       | SOAP          |  91.5136 |  90.1383 | MILO_LW  | 0.000289624 | ***           | final_validation_accuracy |
| SGD           | ADAMW         |  91.6667 |  89.2099 | SGD      | 4.77772e-05 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  91.6667 |  92.0321 | ADAGRAD  | 0.00345784  | **            | final_validation_accuracy |
| SGD           | ADEMAMIX      |  91.6667 |  88.1605 | SGD      | 0.0110047   | *             | final_validation_accuracy |
| SGD           | SOAP          |  91.6667 |  90.1383 | SGD      | 0.00022851  | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  89.2099 |  92.0321 | ADAGRAD  | 4.64857e-06 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  89.2099 |  88.1605 | ADAMW    | 0.254479    |               | final_validation_accuracy |
| ADAMW         | SOAP          |  89.2099 |  90.1383 | SOAP     | 0.00185406  | **            | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  92.0321 |  88.1605 | ADAGRAD  | 0.00762805  | **            | final_validation_accuracy |
| ADAGRAD       | SOAP          |  92.0321 |  90.1383 | ADAGRAD  | 1.92814e-05 | ***           | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  88.1605 |  90.1383 | SOAP     | 0.0642717   |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9135 | 0.0010 | 0.0004 | 0.9123 | 0.9147 |
| MILO_LW | 0.9144 | 0.0009 | 0.0004 | 0.9134 | 0.9155 |
| SGD | 0.9159 | 0.0007 | 0.0003 | 0.9151 | 0.9167 |
| ADAMW | 0.8912 | 0.0033 | 0.0015 | 0.8871 | 0.8953 |
| ADAGRAD | 0.9196 | 0.0016 | 0.0007 | 0.9177 | 0.9215 |
| ADEMAMIX | 0.8799 | 0.0188 | 0.0084 | 0.8566 | 0.9032 |
| SOAP | 0.9006 | 0.0032 | 0.0014 | 0.8966 | 0.9046 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.913474 | 0.914448 | MILO_LW  | 0.131677    |               | final_validation_f1_score |
| MILO          | SGD           | 0.913474 | 0.91591  | SGD      | 0.00210147  | **            | final_validation_f1_score |
| MILO          | ADAMW         | 0.913474 | 0.891225 | MILO     | 4.57835e-05 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.913474 | 0.919605 | ADAGRAD  | 0.000184102 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.913474 | 0.879886 | MILO     | 0.0160725   | *             | final_validation_f1_score |
| MILO          | SOAP          | 0.913474 | 0.900592 | MILO     | 0.000462564 | ***           | final_validation_f1_score |
| MILO_LW       | SGD           | 0.914448 | 0.91591  | SGD      | 0.0198303   | *             | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.914448 | 0.891225 | MILO_LW  | 4.24528e-05 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.914448 | 0.919605 | ADAGRAD  | 0.000548242 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.914448 | 0.879886 | MILO_LW  | 0.014615    | *             | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.914448 | 0.900592 | MILO_LW  | 0.000362335 | ***           | final_validation_f1_score |
| SGD           | ADAMW         | 0.91591  | 0.891225 | SGD      | 4.51806e-05 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.91591  | 0.919605 | ADAGRAD  | 0.00373207  | **            | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.91591  | 0.879886 | SGD      | 0.0127309   | *             | final_validation_f1_score |
| SGD           | SOAP          | 0.91591  | 0.900592 | SGD      | 0.000297556 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.891225 | 0.919605 | ADAGRAD  | 3.5158e-06  | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.891225 | 0.879886 | ADAMW    | 0.250683    |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.891225 | 0.900592 | SOAP     | 0.0018258   | **            | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.919605 | 0.879886 | ADAGRAD  | 0.00892988  | **            | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.919605 | 0.900592 | ADAGRAD  | 2.60226e-05 | ***           | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.879886 | 0.900592 | SOAP     | 0.0685587   |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9921 | 0.0000 | 0.0000 | 0.9920 | 0.9921 |
| MILO_LW | 0.9921 | 0.0001 | 0.0000 | 0.9920 | 0.9922 |
| SGD | 0.9927 | 0.0001 | 0.0000 | 0.9926 | 0.9927 |
| ADAMW | 0.9903 | 0.0003 | 0.0001 | 0.9899 | 0.9907 |
| ADAGRAD | 0.9931 | 0.0000 | 0.0000 | 0.9931 | 0.9931 |
| ADEMAMIX | 0.9899 | 0.0005 | 0.0002 | 0.9893 | 0.9906 |
| SOAP | 0.9908 | 0.0003 | 0.0001 | 0.9905 | 0.9911 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.992081 | 0.992072 | MILO     | 0.788362    |               | final_validation_auc |
| MILO          | SGD           | 0.992081 | 0.992665 | SGD      | 7.09617e-07 | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.992081 | 0.990261 | MILO     | 0.000204115 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.992081 | 0.99309  | ADAGRAD  | 3.83983e-11 | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.992081 | 0.989929 | MILO     | 0.000797067 | ***           | final_validation_auc |
| MILO          | SOAP          | 0.992081 | 0.990777 | MILO     | 0.000277167 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.992072 | 0.992665 | SGD      | 4.73961e-07 | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.992072 | 0.990261 | MILO_LW  | 0.00015454  | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.992072 | 0.99309  | ADAGRAD  | 2.09467e-07 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.992072 | 0.989929 | MILO_LW  | 0.000744744 | ***           | final_validation_auc |
| MILO_LW       | SOAP          | 0.992072 | 0.990777 | MILO_LW  | 0.000184702 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.992665 | 0.990261 | SGD      | 5.13148e-05 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.992665 | 0.99309  | ADAGRAD  | 8.29784e-06 | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.992665 | 0.989929 | SGD      | 0.000289395 | ***           | final_validation_auc |
| SGD           | SOAP          | 0.992665 | 0.990777 | SGD      | 4.05623e-05 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.990261 | 0.99309  | ADAGRAD  | 3.57104e-05 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.990261 | 0.989929 | ADAMW    | 0.274122    |               | final_validation_auc |
| ADAMW         | SOAP          | 0.990261 | 0.990777 | SOAP     | 0.0238387   | *             | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.99309  | 0.989929 | ADAGRAD  | 0.000178213 | ***           | final_validation_auc |
| ADAGRAD       | SOAP          | 0.99309  | 0.990777 | ADAGRAD  | 2.83009e-05 | ***           | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.989929 | 0.990777 | SOAP     | 0.019228    | *             | final_validation_auc |

