# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.1914 | 0.0056 | 0.0025 | 0.1844 | 0.1984 |
| MILO_LW | 0.1400 | 0.0059 | 0.0026 | 0.1327 | 0.1472 |
| MUON | 0.2296 | 0.0155 | 0.0069 | 0.2103 | 0.2489 |
| ADALAYER | 0.2394 | 0.0119 | 0.0053 | 0.2246 | 0.2542 |
| ADAM_MINI | 0.6585 | 0.1318 | 0.0589 | 0.4949 | 0.8221 |
| SGD | 0.1115 | 0.0023 | 0.0010 | 0.1086 | 0.1144 |
| ADAMW | 0.3117 | 0.0336 | 0.0150 | 0.2700 | 0.3535 |
| ADAGRAD | 0.2050 | 0.0132 | 0.0059 | 0.1886 | 0.2214 |
| NOVOGRAD | 0.0993 | 0.0116 | 0.0052 | 0.0848 | 0.1137 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |    Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|----------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.191389 | 0.139962  | MILO_LW  | 6.11071e-07 | ***           | final_validation_loss |
| MILO          | MUON          | 0.191389 | 0.22959   | MILO     | 0.00346934  | **            | final_validation_loss |
| MILO          | ADALAYER      | 0.191389 | 0.239375  | MILO     | 0.000239459 | ***           | final_validation_loss |
| MILO          | ADAM_MINI     | 0.191389 | 0.658527  | MILO     | 0.00135476  | **            | final_validation_loss |
| MILO          | SGD           | 0.191389 | 0.111482  | SGD      | 4.26194e-07 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.191389 | 0.31173   | MILO     | 0.00110676  | **            | final_validation_loss |
| MILO          | ADAGRAD       | 0.191389 | 0.205007  | MILO     | 0.0835824   |               | final_validation_loss |
| MILO          | NOVOGRAD      | 0.191389 | 0.0992502 | NOVOGRAD | 5.33294e-06 | ***           | final_validation_loss |
| MILO_LW       | MUON          | 0.139962 | 0.22959   | MILO_LW  | 5.89195e-05 | ***           | final_validation_loss |
| MILO_LW       | ADALAYER      | 0.139962 | 0.239375  | MILO_LW  | 3.79138e-06 | ***           | final_validation_loss |
| MILO_LW       | ADAM_MINI     | 0.139962 | 0.658527  | MILO_LW  | 0.000907018 | ***           | final_validation_loss |
| MILO_LW       | SGD           | 0.139962 | 0.111482  | SGD      | 0.000124668 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         | 0.139962 | 0.31173   | MILO_LW  | 0.000255901 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.139962 | 0.205007  | MILO_LW  | 9.46581e-05 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.139962 | 0.0992502 | NOVOGRAD | 0.000454425 | ***           | final_validation_loss |
| MUON          | ADALAYER      | 0.22959  | 0.239375  | MUON     | 0.297966    |               | final_validation_loss |
| MUON          | ADAM_MINI     | 0.22959  | 0.658527  | MUON     | 0.00174788  | **            | final_validation_loss |
| MUON          | SGD           | 0.22959  | 0.111482  | SGD      | 5.3412e-05  | ***           | final_validation_loss |
| MUON          | ADAMW         | 0.22959  | 0.31173   | MUON     | 0.00305189  | **            | final_validation_loss |
| MUON          | ADAGRAD       | 0.22959  | 0.205007  | ADAGRAD  | 0.0278664   | *             | final_validation_loss |
| MUON          | NOVOGRAD      | 0.22959  | 0.0992502 | NOVOGRAD | 8.03211e-07 | ***           | final_validation_loss |
| ADALAYER      | ADAM_MINI     | 0.239375 | 0.658527  | ADALAYER | 0.00197156  | **            | final_validation_loss |
| ADALAYER      | SGD           | 0.239375 | 0.111482  | SGD      | 1.02865e-05 | ***           | final_validation_loss |
| ADALAYER      | ADAMW         | 0.239375 | 0.31173   | ADALAYER | 0.00621976  | **            | final_validation_loss |
| ADALAYER      | ADAGRAD       | 0.239375 | 0.205007  | ADAGRAD  | 0.00262436  | **            | final_validation_loss |
| ADALAYER      | NOVOGRAD      | 0.239375 | 0.0992502 | NOVOGRAD | 6.60328e-08 | ***           | final_validation_loss |
| ADAM_MINI     | SGD           | 0.658527 | 0.111482  | SGD      | 0.000747252 | ***           | final_validation_loss |
| ADAM_MINI     | ADAMW         | 0.658527 | 0.31173   | ADAMW    | 0.00320501  | **            | final_validation_loss |
| ADAM_MINI     | ADAGRAD       | 0.658527 | 0.205007  | ADAGRAD  | 0.00144134  | **            | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      | 0.658527 | 0.0992502 | NOVOGRAD | 0.000648048 | ***           | final_validation_loss |
| SGD           | ADAMW         | 0.111482 | 0.31173   | SGD      | 0.000174856 | ***           | final_validation_loss |
| SGD           | ADAGRAD       | 0.111482 | 0.205007  | SGD      | 6.59138e-05 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 0.111482 | 0.0992502 | NOVOGRAD | 0.0773356   |               | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.31173  | 0.205007  | ADAGRAD  | 0.00101907  | **            | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.31173  | 0.0992502 | NOVOGRAD | 4.55918e-05 | ***           | final_validation_loss |
| ADAGRAD       | NOVOGRAD      | 0.205007 | 0.0992502 | NOVOGRAD | 1.05317e-06 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 97.4741 | 0.0744 | 0.0333 | 97.3817 | 97.5664 |
| MILO_LW | 97.0099 | 0.2255 | 0.1009 | 96.7299 | 97.2899 |
| MUON | 96.5383 | 0.1730 | 0.0774 | 96.3235 | 96.7530 |
| ADALAYER | 93.8346 | 0.4457 | 0.1993 | 93.2812 | 94.3880 |
| ADAM_MINI | 85.3432 | 3.4940 | 1.5626 | 81.0049 | 89.6816 |
| SGD | 97.0222 | 0.1613 | 0.0721 | 96.8219 | 97.2225 |
| ADAMW | 93.7654 | 0.5639 | 0.2522 | 93.0652 | 94.4656 |
| ADAGRAD | 94.4173 | 0.3907 | 0.1747 | 93.9321 | 94.9024 |
| NOVOGRAD | 97.0519 | 0.2517 | 0.1126 | 96.7393 | 97.3644 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  97.4741 |  97.0099 | MILO     | 0.00770322  | **            | final_validation_accuracy |
| MILO          | MUON          |  97.4741 |  96.5383 | MILO     | 6.11659e-05 | ***           | final_validation_accuracy |
| MILO          | ADALAYER      |  97.4741 |  93.8346 | MILO     | 3.74007e-05 | ***           | final_validation_accuracy |
| MILO          | ADAM_MINI     |  97.4741 |  85.3432 | MILO     | 0.0014795   | **            | final_validation_accuracy |
| MILO          | SGD           |  97.4741 |  97.0222 | MILO     | 0.00158294  | **            | final_validation_accuracy |
| MILO          | ADAMW         |  97.4741 |  93.7654 | MILO     | 0.000103009 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  97.4741 |  94.4173 | MILO     | 4.05094e-05 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  97.4741 |  97.0519 | MILO     | 0.0173784   | *             | final_validation_accuracy |
| MILO_LW       | MUON          |  97.0099 |  96.5383 | MILO_LW  | 0.00668212  | **            | final_validation_accuracy |
| MILO_LW       | ADALAYER      |  97.0099 |  93.8346 | MILO_LW  | 8.42557e-06 | ***           | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  97.0099 |  85.3432 | MILO_LW  | 0.00167742  | **            | final_validation_accuracy |
| MILO_LW       | SGD           |  97.0099 |  97.0222 | SGD      | 0.923391    |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  97.0099 |  93.7654 | MILO_LW  | 5.27615e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  97.0099 |  94.4173 | MILO_LW  | 8.29498e-06 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  97.0099 |  97.0519 | NOVOGRAD | 0.788352    |               | final_validation_accuracy |
| MUON          | ADALAYER      |  96.5383 |  93.8346 | MUON     | 4.32665e-05 | ***           | final_validation_accuracy |
| MUON          | ADAM_MINI     |  96.5383 |  85.3432 | MUON     | 0.00198107  | **            | final_validation_accuracy |
| MUON          | SGD           |  96.5383 |  97.0222 | SGD      | 0.00183557  | **            | final_validation_accuracy |
| MUON          | ADAMW         |  96.5383 |  93.7654 | MUON     | 0.000182061 | ***           | final_validation_accuracy |
| MUON          | ADAGRAD       |  96.5383 |  94.4173 | MUON     | 5.60864e-05 | ***           | final_validation_accuracy |
| MUON          | NOVOGRAD      |  96.5383 |  97.0519 | NOVOGRAD | 0.00691172  | **            | final_validation_accuracy |
| ADALAYER      | ADAM_MINI     |  93.8346 |  85.3432 | ADALAYER | 0.00522568  | **            | final_validation_accuracy |
| ADALAYER      | SGD           |  93.8346 |  97.0222 | SGD      | 2.24993e-05 | ***           | final_validation_accuracy |
| ADALAYER      | ADAMW         |  93.8346 |  93.7654 | ADALAYER | 0.835373    |               | final_validation_accuracy |
| ADALAYER      | ADAGRAD       |  93.8346 |  94.4173 | ADAGRAD  | 0.0597143   |               | final_validation_accuracy |
| ADALAYER      | NOVOGRAD      |  93.8346 |  97.0519 | NOVOGRAD | 5.3056e-06  | ***           | final_validation_accuracy |
| ADAM_MINI     | SGD           |  85.3432 |  97.0222 | SGD      | 0.00169113  | **            | final_validation_accuracy |
| ADAM_MINI     | ADAMW         |  85.3432 |  93.7654 | ADAMW    | 0.00519713  | **            | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  85.3432 |  94.4173 | ADAGRAD  | 0.00414659  | **            | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  85.3432 |  97.0519 | NOVOGRAD | 0.00164456  | **            | final_validation_accuracy |
| SGD           | ADAMW         |  97.0222 |  93.7654 | SGD      | 9.64467e-05 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  97.0222 |  94.4173 | SGD      | 2.28256e-05 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  97.0222 |  97.0519 | NOVOGRAD | 0.831111    |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  93.7654 |  94.4173 | ADAGRAD  | 0.0705741   |               | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  93.7654 |  97.0519 | NOVOGRAD | 3.7623e-05  | ***           | final_validation_accuracy |
| ADAGRAD       | NOVOGRAD      |  94.4173 |  97.0519 | NOVOGRAD | 5.364e-06   | ***           | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9746 | 0.0008 | 0.0003 | 0.9736 | 0.9755 |
| MILO_LW | 0.9699 | 0.0023 | 0.0010 | 0.9670 | 0.9728 |
| MUON | 0.9652 | 0.0017 | 0.0008 | 0.9631 | 0.9673 |
| ADALAYER | 0.9380 | 0.0046 | 0.0020 | 0.9323 | 0.9437 |
| ADAM_MINI | 0.8573 | 0.0400 | 0.0179 | 0.8077 | 0.9070 |
| SGD | 0.9702 | 0.0016 | 0.0007 | 0.9682 | 0.9722 |
| ADAMW | 0.9374 | 0.0058 | 0.0026 | 0.9301 | 0.9446 |
| ADAGRAD | 0.9439 | 0.0039 | 0.0018 | 0.9391 | 0.9488 |
| NOVOGRAD | 0.9704 | 0.0025 | 0.0011 | 0.9673 | 0.9735 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.974592 | 0.969912 | MILO     | 0.00863013  | **            | final_validation_f1_score |
| MILO          | MUON          | 0.974592 | 0.965192 | MILO     | 5.35683e-05 | ***           | final_validation_f1_score |
| MILO          | ADALAYER      | 0.974592 | 0.938007 | MILO     | 4.09379e-05 | ***           | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.974592 | 0.857315 | MILO     | 0.00278811  | **            | final_validation_f1_score |
| MILO          | SGD           | 0.974592 | 0.970217 | MILO     | 0.0018679   | **            | final_validation_f1_score |
| MILO          | ADAMW         | 0.974592 | 0.937373 | MILO     | 0.000117359 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.974592 | 0.943943 | MILO     | 4.02406e-05 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.974592 | 0.97041  | MILO     | 0.0168724   | *             | final_validation_f1_score |
| MILO_LW       | MUON          | 0.969912 | 0.965192 | MILO_LW  | 0.00763478  | **            | final_validation_f1_score |
| MILO_LW       | ADALAYER      | 0.969912 | 0.938007 | MILO_LW  | 9.25055e-06 | ***           | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.969912 | 0.857315 | MILO_LW  | 0.00319473  | **            | final_validation_f1_score |
| MILO_LW       | SGD           | 0.969912 | 0.970217 | SGD      | 0.816948    |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.969912 | 0.937373 | MILO_LW  | 6.18503e-05 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.969912 | 0.943943 | MILO_LW  | 7.68132e-06 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.969912 | 0.97041  | NOVOGRAD | 0.752045    |               | final_validation_f1_score |
| MUON          | ADALAYER      | 0.965192 | 0.938007 | MUON     | 5.17952e-05 | ***           | final_validation_f1_score |
| MUON          | ADAM_MINI     | 0.965192 | 0.857315 | MUON     | 0.00377064  | **            | final_validation_f1_score |
| MUON          | SGD           | 0.965192 | 0.970217 | SGD      | 0.00144321  | **            | final_validation_f1_score |
| MUON          | ADAMW         | 0.965192 | 0.937373 | MUON     | 0.000221684 | ***           | final_validation_f1_score |
| MUON          | ADAGRAD       | 0.965192 | 0.943943 | MUON     | 5.86108e-05 | ***           | final_validation_f1_score |
| MUON          | NOVOGRAD      | 0.965192 | 0.97041  | NOVOGRAD | 0.00593582  | **            | final_validation_f1_score |
| ADALAYER      | ADAM_MINI     | 0.938007 | 0.857315 | ADALAYER | 0.0103229   | *             | final_validation_f1_score |
| ADALAYER      | SGD           | 0.938007 | 0.970217 | SGD      | 2.56129e-05 | ***           | final_validation_f1_score |
| ADALAYER      | ADAMW         | 0.938007 | 0.937373 | ADALAYER | 0.853642    |               | final_validation_f1_score |
| ADALAYER      | ADAGRAD       | 0.938007 | 0.943943 | ADAGRAD  | 0.0596065   |               | final_validation_f1_score |
| ADALAYER      | NOVOGRAD      | 0.938007 | 0.97041  | NOVOGRAD | 6.89778e-06 | ***           | final_validation_f1_score |
| ADAM_MINI     | SGD           | 0.857315 | 0.970217 | SGD      | 0.00319011  | **            | final_validation_f1_score |
| ADAM_MINI     | ADAMW         | 0.857315 | 0.937373 | ADAMW    | 0.0103827   | *             | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.857315 | 0.943943 | ADAGRAD  | 0.00811001  | **            | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.857315 | 0.97041  | NOVOGRAD | 0.00313621  | **            | final_validation_f1_score |
| SGD           | ADAMW         | 0.970217 | 0.937373 | SGD      | 0.000113043 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.970217 | 0.943943 | SGD      | 2.2382e-05  | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.970217 | 0.97041  | NOVOGRAD | 0.888437    |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.937373 | 0.943943 | ADAGRAD  | 0.0751562   |               | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.937373 | 0.97041  | NOVOGRAD | 4.98431e-05 | ***           | final_validation_f1_score |
| ADAGRAD       | NOVOGRAD      | 0.943943 | 0.97041  | NOVOGRAD | 5.69235e-06 | ***           | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9994 | 0.0000 | 0.0000 | 0.9993 | 0.9994 |
| MILO_LW | 0.9991 | 0.0002 | 0.0001 | 0.9989 | 0.9993 |
| MUON | 0.9989 | 0.0002 | 0.0001 | 0.9987 | 0.9991 |
| ADALAYER | 0.9963 | 0.0005 | 0.0002 | 0.9957 | 0.9970 |
| ADAM_MINI | 0.9764 | 0.0102 | 0.0046 | 0.9637 | 0.9891 |
| SGD | 0.9991 | 0.0001 | 0.0000 | 0.9990 | 0.9991 |
| ADAMW | 0.9958 | 0.0005 | 0.0002 | 0.9952 | 0.9964 |
| ADAGRAD | 0.9971 | 0.0004 | 0.0002 | 0.9966 | 0.9975 |
| NOVOGRAD | 0.9994 | 0.0001 | 0.0001 | 0.9992 | 0.9995 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.999373 | 0.999086 | MILO     | 0.0148742   | *             | final_validation_auc |
| MILO          | MUON          | 0.999373 | 0.998934 | MILO     | 0.00229103  | **            | final_validation_auc |
| MILO          | ADALAYER      | 0.999373 | 0.99634  | MILO     | 0.00018245  | ***           | final_validation_auc |
| MILO          | ADAM_MINI     | 0.999373 | 0.976401 | MILO     | 0.00731586  | **            | final_validation_auc |
| MILO          | SGD           | 0.999373 | 0.999073 | MILO     | 3.0215e-05  | ***           | final_validation_auc |
| MILO          | ADAMW         | 0.999373 | 0.995779 | MILO     | 6.41386e-05 | ***           | final_validation_auc |
| MILO          | ADAGRAD       | 0.999373 | 0.99705  | MILO     | 0.00018806  | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.999373 | 0.999355 | MILO     | 0.750788    |               | final_validation_auc |
| MILO_LW       | MUON          | 0.999086 | 0.998934 | MILO_LW  | 0.160385    |               | final_validation_auc |
| MILO_LW       | ADALAYER      | 0.999086 | 0.99634  | MILO_LW  | 0.000120187 | ***           | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.999086 | 0.976401 | MILO_LW  | 0.00764234  | **            | final_validation_auc |
| MILO_LW       | SGD           | 0.999086 | 0.999073 | MILO_LW  | 0.875094    |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.999086 | 0.995779 | MILO_LW  | 2.73156e-05 | ***           | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.999086 | 0.99705  | MILO_LW  | 9.52968e-05 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.999086 | 0.999355 | NOVOGRAD | 0.0181288   | *             | final_validation_auc |
| MUON          | ADALAYER      | 0.998934 | 0.99634  | MUON     | 0.000171112 | ***           | final_validation_auc |
| MUON          | ADAM_MINI     | 0.998934 | 0.976401 | MUON     | 0.00782698  | **            | final_validation_auc |
| MUON          | SGD           | 0.998934 | 0.999073 | SGD      | 0.108079    |               | final_validation_auc |
| MUON          | ADAMW         | 0.998934 | 0.995779 | MUON     | 3.91628e-05 | ***           | final_validation_auc |
| MUON          | ADAGRAD       | 0.998934 | 0.99705  | MUON     | 0.000158893 | ***           | final_validation_auc |
| MUON          | NOVOGRAD      | 0.998934 | 0.999355 | NOVOGRAD | 0.00133841  | **            | final_validation_auc |
| ADALAYER      | ADAM_MINI     | 0.99634  | 0.976401 | ADALAYER | 0.0118979   | *             | final_validation_auc |
| ADALAYER      | SGD           | 0.99634  | 0.999073 | SGD      | 0.000255624 | ***           | final_validation_auc |
| ADALAYER      | ADAMW         | 0.99634  | 0.995779 | ADALAYER | 0.109257    |               | final_validation_auc |
| ADALAYER      | ADAGRAD       | 0.99634  | 0.99705  | ADAGRAD  | 0.0420584   | *             | final_validation_auc |
| ADALAYER      | NOVOGRAD      | 0.99634  | 0.999355 | NOVOGRAD | 0.000117637 | ***           | final_validation_auc |
| ADAM_MINI     | SGD           | 0.976401 | 0.999073 | SGD      | 0.00766303  | **            | final_validation_auc |
| ADAM_MINI     | ADAMW         | 0.976401 | 0.995779 | ADAMW    | 0.0131299   | *             | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.976401 | 0.99705  | ADAGRAD  | 0.0105753   | *             | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.976401 | 0.999355 | NOVOGRAD | 0.00733333  | **            | final_validation_auc |
| SGD           | ADAMW         | 0.999073 | 0.995779 | SGD      | 8.14916e-05 | ***           | final_validation_auc |
| SGD           | ADAGRAD       | 0.999073 | 0.99705  | SGD      | 0.000289453 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.999073 | 0.999355 | NOVOGRAD | 0.00315881  | **            | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.995779 | 0.99705  | ADAGRAD  | 0.00182632  | **            | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.995779 | 0.999355 | NOVOGRAD | 3.36743e-05 | ***           | final_validation_auc |
| ADAGRAD       | NOVOGRAD      | 0.99705  | 0.999355 | NOVOGRAD | 9.31147e-05 | ***           | final_validation_auc |

