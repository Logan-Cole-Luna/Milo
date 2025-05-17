# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.2259 | 0.0182 | 0.0081 | 0.2034 | 0.2485 |
| MILO_LW | 0.1507 | 0.0102 | 0.0045 | 0.1381 | 0.1634 |
| SGD | 0.0806 | 0.0026 | 0.0012 | 0.0774 | 0.0838 |
| ADAMW | 0.3011 | 0.0235 | 0.0105 | 0.2719 | 0.3303 |
| ADAGRAD | 0.0882 | 0.0048 | 0.0021 | 0.0823 | 0.0941 |
| NOVOGRAD | 0.1028 | 0.0078 | 0.0035 | 0.0931 | 0.1124 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |    Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|----------:|----------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.225943  | 0.150747  | MILO_LW  | 0.000151986 | ***           | final_validation_loss |
| MILO          | SGD           | 0.225943  | 0.0806139 | SGD      | 4.4526e-05  | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.225943  | 0.301111  | MILO     | 0.000599014 | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.225943  | 0.0881714 | ADAGRAD  | 3.20264e-05 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      | 0.225943  | 0.102758  | NOVOGRAD | 1.89811e-05 | ***           | final_validation_loss |
| MILO_LW       | SGD           | 0.150747  | 0.0806139 | SGD      | 5.0823e-05  | ***           | final_validation_loss |
| MILO_LW       | ADAMW         | 0.150747  | 0.301111  | MILO_LW  | 2.51741e-05 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.150747  | 0.0881714 | ADAGRAD  | 2.45215e-05 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.150747  | 0.102758  | NOVOGRAD | 4.59675e-05 | ***           | final_validation_loss |
| SGD           | ADAMW         | 0.0806139 | 0.301111  | SGD      | 2.59795e-05 | ***           | final_validation_loss |
| SGD           | ADAGRAD       | 0.0806139 | 0.0881714 | SGD      | 0.0198726   | *             | final_validation_loss |
| SGD           | NOVOGRAD      | 0.0806139 | 0.102758  | SGD      | 0.00191629  | **            | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.301111  | 0.0881714 | ADAGRAD  | 2.0569e-05  | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.301111  | 0.102758  | NOVOGRAD | 1.27102e-05 | ***           | final_validation_loss |
| ADAGRAD       | NOVOGRAD      | 0.0881714 | 0.102758  | ADAGRAD  | 0.00973201  | **            | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 97.3210 | 0.1654 | 0.0740 | 97.1156 | 97.5264 |
| MILO_LW | 96.8198 | 0.1326 | 0.0593 | 96.6552 | 96.9844 |
| SGD | 97.4889 | 0.0596 | 0.0267 | 97.4149 | 97.5629 |
| ADAMW | 92.9728 | 0.6616 | 0.2959 | 92.1513 | 93.7944 |
| ADAGRAD | 97.2988 | 0.0650 | 0.0291 | 97.2181 | 97.3794 |
| NOVOGRAD | 96.9259 | 0.1142 | 0.0511 | 96.7842 | 97.0677 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  97.321  |  96.8198 | MILO     | 0.000859848 | ***           | final_validation_accuracy |
| MILO          | SGD           |  97.321  |  97.4889 | SGD      | 0.0855765   |               | final_validation_accuracy |
| MILO          | ADAMW         |  97.321  |  92.9728 | MILO     | 6.47594e-05 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  97.321  |  97.2988 | MILO     | 0.790543    |               | final_validation_accuracy |
| MILO          | NOVOGRAD      |  97.321  |  96.9259 | MILO     | 0.00306135  | **            | final_validation_accuracy |
| MILO_LW       | SGD           |  96.8198 |  97.4889 | SGD      | 7.95883e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADAMW         |  96.8198 |  92.9728 | MILO_LW  | 0.000136568 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  96.8198 |  97.2988 | ADAGRAD  | 0.000400961 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  96.8198 |  96.9259 | NOVOGRAD | 0.212593    |               | final_validation_accuracy |
| SGD           | ADAMW         |  97.4889 |  92.9728 | SGD      | 9.81285e-05 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  97.4889 |  97.2988 | SGD      | 0.00134598  | **            | final_validation_accuracy |
| SGD           | NOVOGRAD      |  97.4889 |  96.9259 | SGD      | 6.39821e-05 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  92.9728 |  97.2988 | ADAGRAD  | 0.000114652 | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  92.9728 |  96.9259 | NOVOGRAD | 0.000134631 | ***           | final_validation_accuracy |
| ADAGRAD       | NOVOGRAD      |  97.2988 |  96.9259 | ADAGRAD  | 0.000574445 | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9731 | 0.0017 | 0.0008 | 0.9710 | 0.9752 |
| MILO_LW | 0.9680 | 0.0013 | 0.0006 | 0.9664 | 0.9697 |
| SGD | 0.9748 | 0.0006 | 0.0003 | 0.9741 | 0.9756 |
| ADAMW | 0.9299 | 0.0063 | 0.0028 | 0.9220 | 0.9377 |
| ADAGRAD | 0.9729 | 0.0007 | 0.0003 | 0.9721 | 0.9738 |
| NOVOGRAD | 0.9692 | 0.0012 | 0.0005 | 0.9677 | 0.9706 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.973116 | 0.968024 | MILO     | 0.000910593 | ***           | final_validation_f1_score |
| MILO          | SGD           | 0.973116 | 0.97483  | SGD      | 0.0870775   |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.973116 | 0.929862 | MILO     | 4.90212e-05 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.973116 | 0.972917 | MILO     | 0.817466    |               | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.973116 | 0.969182 | MILO     | 0.00360885  | **            | final_validation_f1_score |
| MILO_LW       | SGD           | 0.968024 | 0.97483  | SGD      | 7.95343e-05 | ***           | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.968024 | 0.929862 | MILO_LW  | 0.00011074  | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.968024 | 0.972917 | ADAGRAD  | 0.000359347 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.968024 | 0.969182 | NOVOGRAD | 0.184533    |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.97483  | 0.929862 | SGD      | 8.21883e-05 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.97483  | 0.972917 | SGD      | 0.00174726  | **            | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.97483  | 0.969182 | SGD      | 7.13028e-05 | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.929862 | 0.972917 | ADAGRAD  | 9.46427e-05 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.929862 | 0.969182 | NOVOGRAD | 0.000109223 | ***           | final_validation_f1_score |
| ADAGRAD       | NOVOGRAD      | 0.972917 | 0.969182 | ADAGRAD  | 0.000605984 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9993 | 0.0001 | 0.0000 | 0.9992 | 0.9994 |
| MILO_LW | 0.9990 | 0.0000 | 0.0000 | 0.9989 | 0.9990 |
| SGD | 0.9995 | 0.0000 | 0.0000 | 0.9994 | 0.9995 |
| ADAMW | 0.9949 | 0.0006 | 0.0003 | 0.9941 | 0.9957 |
| ADAGRAD | 0.9994 | 0.0001 | 0.0000 | 0.9993 | 0.9995 |
| NOVOGRAD | 0.9993 | 0.0001 | 0.0000 | 0.9991 | 0.9994 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.999313 | 0.998967 | MILO     | 9.0835e-06  | ***           | final_validation_auc |
| MILO          | SGD           | 0.999313 | 0.999478 | SGD      | 0.00152514  | **            | final_validation_auc |
| MILO          | ADAMW         | 0.999313 | 0.994919 | MILO     | 8.57084e-05 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.999313 | 0.999391 | ADAGRAD  | 0.11668     |               | final_validation_auc |
| MILO          | NOVOGRAD      | 0.999313 | 0.999277 | MILO     | 0.528498    |               | final_validation_auc |
| MILO_LW       | SGD           | 0.998967 | 0.999478 | SGD      | 7.6625e-07  | ***           | final_validation_auc |
| MILO_LW       | ADAMW         | 0.998967 | 0.994919 | MILO_LW  | 0.000122329 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.998967 | 0.999391 | ADAGRAD  | 2.68048e-05 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.998967 | 0.999277 | NOVOGRAD | 0.0011452   | **            | final_validation_auc |
| SGD           | ADAMW         | 0.999478 | 0.994919 | SGD      | 8.02061e-05 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.999478 | 0.999391 | SGD      | 0.0691201   |               | final_validation_auc |
| SGD           | NOVOGRAD      | 0.999478 | 0.999277 | SGD      | 0.0108966   | *             | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.994919 | 0.999391 | ADAGRAD  | 7.32225e-05 | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.994919 | 0.999277 | NOVOGRAD | 7.13922e-05 | ***           | final_validation_auc |
| ADAGRAD       | NOVOGRAD      | 0.999391 | 0.999277 | ADAGRAD  | 0.0905118   |               | final_validation_auc |

