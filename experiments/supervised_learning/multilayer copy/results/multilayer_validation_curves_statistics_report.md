# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.3343 | 0.0183 | 0.0082 | 0.3115 | 0.3570 |
| MILO_LW | 0.2390 | 0.0154 | 0.0069 | 0.2200 | 0.2581 |
| SGD | 0.0980 | 0.0023 | 0.0010 | 0.0952 | 0.1009 |
| ADAMW | 0.4063 | 0.0598 | 0.0268 | 0.3320 | 0.4806 |
| ADAM_MINI | 2.2513 | 0.1181 | 0.0528 | 2.1047 | 2.3979 |
| NOVOGRAD | 0.1112 | 0.0040 | 0.0018 | 0.1063 | 0.1162 |
| ADAGRAD | 0.2231 | 0.0091 | 0.0041 | 0.2118 | 0.2344 |
| ADEMAMIX | 0.3709 | 0.0289 | 0.0129 | 0.3350 | 0.4068 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |    Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|----------:|----------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.334259  | 0.239037  | MILO_LW  | 2.42374e-05 | ***           | final_validation_loss |
| MILO          | SGD           | 0.334259  | 0.0980144 | SGD      | 6.69809e-06 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.334259  | 0.406279  | MILO     | 0.0523869   |               | final_validation_loss |
| MILO          | ADAM_MINI     | 0.334259  | 2.25133   | MILO     | 2.23326e-06 | ***           | final_validation_loss |
| MILO          | NOVOGRAD      | 0.334259  | 0.111235  | NOVOGRAD | 5.20142e-06 | ***           | final_validation_loss |
| MILO          | ADAGRAD       | 0.334259  | 0.223133  | ADAGRAD  | 2.25075e-05 | ***           | final_validation_loss |
| MILO          | ADEMAMIX      | 0.334259  | 0.370884  | MILO     | 0.0490735   | *             | final_validation_loss |
| MILO_LW       | SGD           | 0.239037  | 0.0980144 | SGD      | 2.46606e-05 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         | 0.239037  | 0.406279  | MILO_LW  | 0.00250171  | **            | final_validation_loss |
| MILO_LW       | ADAM_MINI     | 0.239037  | 2.25133   | MILO_LW  | 2.07674e-06 | ***           | final_validation_loss |
| MILO_LW       | NOVOGRAD      | 0.239037  | 0.111235  | NOVOGRAD | 2.13734e-05 | ***           | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.239037  | 0.223133  | ADAGRAD  | 0.0897955   |               | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.239037  | 0.370884  | MILO_LW  | 9.56946e-05 | ***           | final_validation_loss |
| SGD           | ADAMW         | 0.0980144 | 0.406279  | SGD      | 0.00032005  | ***           | final_validation_loss |
| SGD           | ADAM_MINI     | 0.0980144 | 2.25133   | SGD      | 2.14618e-06 | ***           | final_validation_loss |
| SGD           | NOVOGRAD      | 0.0980144 | 0.111235  | SGD      | 0.000534805 | ***           | final_validation_loss |
| SGD           | ADAGRAD       | 0.0980144 | 0.223133  | SGD      | 2.3785e-06  | ***           | final_validation_loss |
| SGD           | ADEMAMIX      | 0.0980144 | 0.370884  | SGD      | 2.72388e-05 | ***           | final_validation_loss |
| ADAMW         | ADAM_MINI     | 0.406279  | 2.25133   | ADAMW    | 8.45833e-08 | ***           | final_validation_loss |
| ADAMW         | NOVOGRAD      | 0.406279  | 0.111235  | NOVOGRAD | 0.00036999  | ***           | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.406279  | 0.223133  | ADAGRAD  | 0.00211215  | **            | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.406279  | 0.370884  | ADEMAMIX | 0.280333    |               | final_validation_loss |
| ADAM_MINI     | NOVOGRAD      | 2.25133   | 0.111235  | NOVOGRAD | 2.1677e-06  | ***           | final_validation_loss |
| ADAM_MINI     | ADAGRAD       | 2.25133   | 0.223133  | ADAGRAD  | 2.45952e-06 | ***           | final_validation_loss |
| ADAM_MINI     | ADEMAMIX      | 2.25133   | 0.370884  | ADEMAMIX | 1.30799e-06 | ***           | final_validation_loss |
| NOVOGRAD      | ADAGRAD       | 0.111235  | 0.223133  | NOVOGRAD | 6.94201e-07 | ***           | final_validation_loss |
| NOVOGRAD      | ADEMAMIX      | 0.111235  | 0.370884  | NOVOGRAD | 2.80436e-05 | ***           | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 0.223133  | 0.370884  | ADAGRAD  | 0.000146224 | ***           | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 97.2370 | 0.1042 | 0.0466 | 97.1077 | 97.3664 |
| MILO_LW | 97.3852 | 0.1045 | 0.0468 | 97.2554 | 97.5150 |
| SGD | 97.2790 | 0.1454 | 0.0650 | 97.0984 | 97.4596 |
| ADAMW | 92.4988 | 0.2456 | 0.1098 | 92.1938 | 92.8037 |
| ADAM_MINI | 12.1111 | 4.2730 | 1.9109 | 6.8055 | 17.4167 |
| NOVOGRAD | 96.5802 | 0.1649 | 0.0738 | 96.3754 | 96.7851 |
| ADAGRAD | 93.8370 | 0.3269 | 0.1462 | 93.4311 | 94.2430 |
| ADEMAMIX | 92.1284 | 0.9601 | 0.4294 | 90.9362 | 93.3206 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  97.237  |  97.3852 | MILO_LW  | 0.0550245   |               | final_validation_accuracy |
| MILO          | SGD           |  97.237  |  97.279  | SGD      | 0.615497    |               | final_validation_accuracy |
| MILO          | ADAMW         |  97.237  |  92.4988 | MILO     | 7.27503e-08 | ***           | final_validation_accuracy |
| MILO          | ADAM_MINI     |  97.237  |  12.1111 | MILO     | 1.50089e-06 | ***           | final_validation_accuracy |
| MILO          | NOVOGRAD      |  97.237  |  96.5802 | MILO     | 0.000160433 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  97.237  |  93.837  | MILO     | 5.05005e-06 | ***           | final_validation_accuracy |
| MILO          | ADEMAMIX      |  97.237  |  92.1284 | MILO     | 0.000256207 | ***           | final_validation_accuracy |
| MILO_LW       | SGD           |  97.3852 |  97.279  | MILO_LW  | 0.225204    |               | final_validation_accuracy |
| MILO_LW       | ADAMW         |  97.3852 |  92.4988 | MILO_LW  | 6.04489e-08 | ***           | final_validation_accuracy |
| MILO_LW       | ADAM_MINI     |  97.3852 |  12.1111 | MILO_LW  | 1.49037e-06 | ***           | final_validation_accuracy |
| MILO_LW       | NOVOGRAD      |  97.3852 |  96.5802 | MILO_LW  | 4.49957e-05 | ***           | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  97.3852 |  93.837  | MILO_LW  | 4.08307e-06 | ***           | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  97.3852 |  92.1284 | MILO_LW  | 0.000228349 | ***           | final_validation_accuracy |
| SGD           | ADAMW         |  97.279  |  92.4988 | SGD      | 7.75627e-09 | ***           | final_validation_accuracy |
| SGD           | ADAM_MINI     |  97.279  |  12.1111 | SGD      | 1.48137e-06 | ***           | final_validation_accuracy |
| SGD           | NOVOGRAD      |  97.279  |  96.5802 | SGD      | 0.000109506 | ***           | final_validation_accuracy |
| SGD           | ADAGRAD       |  97.279  |  93.837  | SGD      | 1.54041e-06 | ***           | final_validation_accuracy |
| SGD           | ADEMAMIX      |  97.279  |  92.1284 | SGD      | 0.000223764 | ***           | final_validation_accuracy |
| ADAMW         | ADAM_MINI     |  92.4988 |  12.1111 | ADAMW    | 1.7912e-06  | ***           | final_validation_accuracy |
| ADAMW         | NOVOGRAD      |  92.4988 |  96.5802 | NOVOGRAD | 9.75018e-09 | ***           | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  92.4988 |  93.837  | ADAGRAD  | 0.00011995  | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  92.4988 |  92.1284 | ADAMW    | 0.445249    |               | final_validation_accuracy |
| ADAM_MINI     | NOVOGRAD      |  12.1111 |  96.5802 | NOVOGRAD | 1.5211e-06  | ***           | final_validation_accuracy |
| ADAM_MINI     | ADAGRAD       |  12.1111 |  93.837  | ADAGRAD  | 1.59561e-06 | ***           | final_validation_accuracy |
| ADAM_MINI     | ADEMAMIX      |  12.1111 |  92.1284 | ADEMAMIX | 7.52549e-07 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADAGRAD       |  96.5802 |  93.837  | NOVOGRAD | 3.30081e-06 | ***           | final_validation_accuracy |
| NOVOGRAD      | ADEMAMIX      |  96.5802 |  92.1284 | NOVOGRAD | 0.000384185 | ***           | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  93.837  |  92.1284 | ADAGRAD  | 0.0134834   | *             | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9722 | 0.0011 | 0.0005 | 0.9709 | 0.9736 |
| MILO_LW | 0.9738 | 0.0011 | 0.0005 | 0.9724 | 0.9751 |
| SGD | 0.9728 | 0.0015 | 0.0007 | 0.9709 | 0.9746 |
| ADAMW | 0.9254 | 0.0025 | 0.0011 | 0.9223 | 0.9285 |
| ADAM_MINI | 0.0376 | 0.0417 | 0.0186 | -0.0142 | 0.0893 |
| NOVOGRAD | 0.9657 | 0.0017 | 0.0007 | 0.9636 | 0.9677 |
| ADAGRAD | 0.9380 | 0.0033 | 0.0015 | 0.9339 | 0.9422 |
| ADEMAMIX | 0.9225 | 0.0087 | 0.0039 | 0.9118 | 0.9333 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |    Mean A |    Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|----------:|----------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.972248  | 0.973754  | MILO_LW  | 0.0573327   |               | final_validation_f1_score |
| MILO          | SGD           | 0.972248  | 0.972759  | SGD      | 0.546444    |               | final_validation_f1_score |
| MILO          | ADAMW         | 0.972248  | 0.925417  | MILO     | 8.38583e-08 | ***           | final_validation_f1_score |
| MILO          | ADAM_MINI     | 0.972248  | 0.0375627 | MILO     | 9.31628e-07 | ***           | final_validation_f1_score |
| MILO          | NOVOGRAD      | 0.972248  | 0.965659  | MILO     | 0.000166492 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.972248  | 0.938033  | MILO     | 5.43603e-06 | ***           | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.972248  | 0.922547  | MILO     | 0.00018589  | ***           | final_validation_f1_score |
| MILO_LW       | SGD           | 0.973754  | 0.972759  | MILO_LW  | 0.260949    |               | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.973754  | 0.925417  | MILO_LW  | 5.88589e-08 | ***           | final_validation_f1_score |
| MILO_LW       | ADAM_MINI     | 0.973754  | 0.0375627 | MILO_LW  | 9.24823e-07 | ***           | final_validation_f1_score |
| MILO_LW       | NOVOGRAD      | 0.973754  | 0.965659  | MILO_LW  | 4.37567e-05 | ***           | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.973754  | 0.938033  | MILO_LW  | 4.06011e-06 | ***           | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.973754  | 0.922547  | MILO_LW  | 0.00016318  | ***           | final_validation_f1_score |
| SGD           | ADAMW         | 0.972759  | 0.925417  | SGD      | 9.41389e-09 | ***           | final_validation_f1_score |
| SGD           | ADAM_MINI     | 0.972759  | 0.0375627 | SGD      | 9.18194e-07 | ***           | final_validation_f1_score |
| SGD           | NOVOGRAD      | 0.972759  | 0.965659  | SGD      | 0.000104687 | ***           | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.972759  | 0.938033  | SGD      | 1.69922e-06 | ***           | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.972759  | 0.922547  | SGD      | 0.000156156 | ***           | final_validation_f1_score |
| ADAMW         | ADAM_MINI     | 0.925417  | 0.0375627 | ADAMW    | 1.07835e-06 | ***           | final_validation_f1_score |
| ADAMW         | NOVOGRAD      | 0.925417  | 0.965659  | NOVOGRAD | 1.2136e-08  | ***           | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.925417  | 0.938033  | ADAGRAD  | 0.000199589 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.925417  | 0.922547  | ADAMW    | 0.51155     |               | final_validation_f1_score |
| ADAM_MINI     | NOVOGRAD      | 0.0375627 | 0.965659  | NOVOGRAD | 9.39657e-07 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADAGRAD       | 0.0375627 | 0.938033  | ADAGRAD  | 9.62166e-07 | ***           | final_validation_f1_score |
| ADAM_MINI     | ADEMAMIX      | 0.0375627 | 0.922547  | ADEMAMIX | 4.95106e-07 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADAGRAD       | 0.965659  | 0.938033  | NOVOGRAD | 3.65843e-06 | ***           | final_validation_f1_score |
| NOVOGRAD      | ADEMAMIX      | 0.965659  | 0.922547  | NOVOGRAD | 0.000273166 | ***           | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.938033  | 0.922547  | ADAGRAD  | 0.0129514   | *             | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9991 | 0.0001 | 0.0000 | 0.9990 | 0.9992 |
| MILO_LW | 0.9993 | 0.0001 | 0.0000 | 0.9993 | 0.9994 |
| SGD | 0.9993 | 0.0000 | 0.0000 | 0.9993 | 0.9994 |
| ADAMW | 0.9931 | 0.0007 | 0.0003 | 0.9923 | 0.9940 |
| ADAM_MINI | 0.5185 | 0.0408 | 0.0182 | 0.4679 | 0.5692 |
| NOVOGRAD | 0.9992 | 0.0001 | 0.0000 | 0.9992 | 0.9993 |
| ADAGRAD | 0.9965 | 0.0003 | 0.0001 | 0.9961 | 0.9968 |
| ADEMAMIX | 0.9932 | 0.0010 | 0.0005 | 0.9920 | 0.9945 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.999072 | 0.999347 | MILO_LW  | 0.000633451 | ***           | final_validation_auc |
| MILO          | SGD           | 0.999072 | 0.999347 | SGD      | 0.00115449  | **            | final_validation_auc |
| MILO          | ADAMW         | 0.999072 | 0.993145 | MILO     | 2.76476e-05 | ***           | final_validation_auc |
| MILO          | ADAM_MINI     | 0.999072 | 0.51852  | MILO     | 1.23596e-05 | ***           | final_validation_auc |
| MILO          | NOVOGRAD      | 0.999072 | 0.999225 | NOVOGRAD | 0.0132902   | *             | final_validation_auc |
| MILO          | ADAGRAD       | 0.999072 | 0.996481 | MILO     | 1.17891e-05 | ***           | final_validation_auc |
| MILO          | ADEMAMIX      | 0.999072 | 0.993242 | MILO     | 0.000210688 | ***           | final_validation_auc |
| MILO_LW       | SGD           | 0.999347 | 0.999347 | MILO_LW  | 0.991736    |               | final_validation_auc |
| MILO_LW       | ADAMW         | 0.999347 | 0.993145 | MILO_LW  | 2.54723e-05 | ***           | final_validation_auc |
| MILO_LW       | ADAM_MINI     | 0.999347 | 0.51852  | MILO_LW  | 1.23319e-05 | ***           | final_validation_auc |
| MILO_LW       | NOVOGRAD      | 0.999347 | 0.999225 | MILO_LW  | 0.0132775   | *             | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.999347 | 0.996481 | MILO_LW  | 1.21954e-05 | ***           | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.999347 | 0.993242 | MILO_LW  | 0.000180987 | ***           | final_validation_auc |
| SGD           | ADAMW         | 0.999347 | 0.993145 | SGD      | 2.8343e-05  | ***           | final_validation_auc |
| SGD           | ADAM_MINI     | 0.999347 | 0.51852  | SGD      | 1.23323e-05 | ***           | final_validation_auc |
| SGD           | NOVOGRAD      | 0.999347 | 0.999225 | SGD      | 0.00426394  | **            | final_validation_auc |
| SGD           | ADAGRAD       | 0.999347 | 0.996481 | SGD      | 2.09439e-05 | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.999347 | 0.993242 | SGD      | 0.000186844 | ***           | final_validation_auc |
| ADAMW         | ADAM_MINI     | 0.993145 | 0.51852  | ADAMW    | 1.2935e-05  | ***           | final_validation_auc |
| ADAMW         | NOVOGRAD      | 0.993145 | 0.999225 | NOVOGRAD | 2.88029e-05 | ***           | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.993145 | 0.996481 | ADAGRAD  | 7.79268e-05 | ***           | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.993145 | 0.993242 | ADEMAMIX | 0.864403    |               | final_validation_auc |
| ADAM_MINI     | NOVOGRAD      | 0.51852  | 0.999225 | NOVOGRAD | 1.23445e-05 | ***           | final_validation_auc |
| ADAM_MINI     | ADAGRAD       | 0.51852  | 0.996481 | ADAGRAD  | 1.26195e-05 | ***           | final_validation_auc |
| ADAM_MINI     | ADEMAMIX      | 0.51852  | 0.993242 | ADEMAMIX | 1.28484e-05 | ***           | final_validation_auc |
| NOVOGRAD      | ADAGRAD       | 0.999225 | 0.996481 | NOVOGRAD | 1.82019e-05 | ***           | final_validation_auc |
| NOVOGRAD      | ADEMAMIX      | 0.999225 | 0.993242 | NOVOGRAD | 0.000198471 | ***           | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.996481 | 0.993242 | ADAGRAD  | 0.00145596  | **            | final_validation_auc |

