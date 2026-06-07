# Statistical Analysis Report

## Methodology

Statistics (mean, standard deviation, standard error, 95% confidence interval) are calculated across multiple runs.
Significance tests (t-test) are performed on the *final epoch's validation metrics* between pairs of optimizers.

### Validation Loss Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.8726 | 0.0212 | 0.0095 | 0.8463 | 0.8989 |
| MILO_LW | 0.8741 | 0.0259 | 0.0116 | 0.8420 | 0.9061 |
| SGD | 1.0272 | 0.0167 | 0.0075 | 1.0065 | 1.0479 |
| ADAMW | 0.8993 | 0.0411 | 0.0184 | 0.8482 | 0.9504 |
| ADAGRAD | 0.9503 | 0.0307 | 0.0138 | 0.9121 | 0.9885 |
| ADEMAMIX | 0.9484 | 0.0996 | 0.0445 | 0.8248 | 1.0720 |
| SOAP | 219992.7194 | 491915.6866 | 219991.3829 | -390801.2787 | 830786.7176 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |        Mean B | Better   |     p-value | Significant   | Metric                |
|:--------------|:--------------|---------:|--------------:|:---------|------------:|:--------------|:----------------------|
| MILO          | MILO_LW       | 0.872574 |      0.874051 | MILO     | 0.923829    |               | final_validation_loss |
| MILO          | SGD           | 0.872574 |      1.02721  | MILO     | 2.07244e-06 | ***           | final_validation_loss |
| MILO          | ADAMW         | 0.872574 |      0.899317 | MILO     | 0.243928    |               | final_validation_loss |
| MILO          | ADAGRAD       | 0.872574 |      0.950302 | MILO     | 0.00224358  | **            | final_validation_loss |
| MILO          | ADEMAMIX      | 0.872574 |      0.948428 | MILO     | 0.165001    |               | final_validation_loss |
| MILO          | SOAP          | 0.872574 | 219993        | MILO     | 0.3739      |               | final_validation_loss |
| MILO_LW       | SGD           | 0.874051 |      1.02721  | MILO_LW  | 1.24606e-05 | ***           | final_validation_loss |
| MILO_LW       | ADAMW         | 0.874051 |      0.899317 | MILO_LW  | 0.284445    |               | final_validation_loss |
| MILO_LW       | ADAGRAD       | 0.874051 |      0.950302 | MILO_LW  | 0.00301228  | **            | final_validation_loss |
| MILO_LW       | ADEMAMIX      | 0.874051 |      0.948428 | MILO_LW  | 0.172718    |               | final_validation_loss |
| MILO_LW       | SOAP          | 0.874051 | 219993        | MILO_LW  | 0.3739      |               | final_validation_loss |
| SGD           | ADAMW         | 1.02721  |      0.899317 | ADAMW    | 0.00108917  | **            | final_validation_loss |
| SGD           | ADAGRAD       | 1.02721  |      0.950302 | ADAGRAD  | 0.00246758  | **            | final_validation_loss |
| SGD           | ADEMAMIX      | 1.02721  |      0.948428 | ADEMAMIX | 0.152048    |               | final_validation_loss |
| SGD           | SOAP          | 1.02721  | 219993        | SGD      | 0.3739      |               | final_validation_loss |
| ADAMW         | ADAGRAD       | 0.899317 |      0.950302 | ADAMW    | 0.0598227   |               | final_validation_loss |
| ADAMW         | ADEMAMIX      | 0.899317 |      0.948428 | ADAMW    | 0.352009    |               | final_validation_loss |
| ADAMW         | SOAP          | 0.899317 | 219993        | ADAMW    | 0.3739      |               | final_validation_loss |
| ADAGRAD       | ADEMAMIX      | 0.950302 |      0.948428 | ADEMAMIX | 0.969551    |               | final_validation_loss |
| ADAGRAD       | SOAP          | 0.950302 | 219993        | ADAGRAD  | 0.3739      |               | final_validation_loss |
| ADEMAMIX      | SOAP          | 0.948428 | 219993        | ADEMAMIX | 0.3739      |               | final_validation_loss |

### Validation Accuracy Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 72.8504 | 0.3820 | 0.1708 | 72.3761 | 73.3247 |
| MILO_LW | 71.8963 | 1.0653 | 0.4764 | 70.5735 | 73.2191 |
| SGD | 68.7970 | 0.2532 | 0.1133 | 68.4826 | 69.1115 |
| ADAMW | 69.5052 | 0.9996 | 0.4470 | 68.2641 | 70.7463 |
| ADAGRAD | 72.7970 | 0.9069 | 0.4056 | 71.6710 | 73.9231 |
| ADEMAMIX | 67.3600 | 3.3392 | 1.4934 | 63.2138 | 71.5062 |
| SOAP | 52.7437 | 33.6108 | 15.0312 | 11.0104 | 94.4770 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       |  72.8504 |  71.8963 | MILO     | 0.117973    |               | final_validation_accuracy |
| MILO          | SGD           |  72.8504 |  68.797  | MILO     | 2.2992e-07  | ***           | final_validation_accuracy |
| MILO          | ADAMW         |  72.8504 |  69.5052 | MILO     | 0.000821171 | ***           | final_validation_accuracy |
| MILO          | ADAGRAD       |  72.8504 |  72.797  | MILO     | 0.907941    |               | final_validation_accuracy |
| MILO          | ADEMAMIX      |  72.8504 |  67.36   | MILO     | 0.0207677   | *             | final_validation_accuracy |
| MILO          | SOAP          |  72.8504 |  52.7437 | MILO     | 0.252005    |               | final_validation_accuracy |
| MILO_LW       | SGD           |  71.8963 |  68.797  | MILO_LW  | 0.00220735  | **            | final_validation_accuracy |
| MILO_LW       | ADAMW         |  71.8963 |  69.5052 | MILO_LW  | 0.00644586  | **            | final_validation_accuracy |
| MILO_LW       | ADAGRAD       |  71.8963 |  72.797  | ADAGRAD  | 0.188866    |               | final_validation_accuracy |
| MILO_LW       | ADEMAMIX      |  71.8963 |  67.36   | MILO_LW  | 0.0356558   | *             | final_validation_accuracy |
| MILO_LW       | SOAP          |  71.8963 |  52.7437 | MILO_LW  | 0.271668    |               | final_validation_accuracy |
| SGD           | ADAMW         |  68.797  |  69.5052 | ADAMW    | 0.191426    |               | final_validation_accuracy |
| SGD           | ADAGRAD       |  68.797  |  72.797  | ADAGRAD  | 0.000332733 | ***           | final_validation_accuracy |
| SGD           | ADEMAMIX      |  68.797  |  67.36   | SGD      | 0.391052    |               | final_validation_accuracy |
| SGD           | SOAP          |  68.797  |  52.7437 | SGD      | 0.345695    |               | final_validation_accuracy |
| ADAMW         | ADAGRAD       |  69.5052 |  72.797  | ADAGRAD  | 0.000625649 | ***           | final_validation_accuracy |
| ADAMW         | ADEMAMIX      |  69.5052 |  67.36   | ADAMW    | 0.230582    |               | final_validation_accuracy |
| ADAMW         | SOAP          |  69.5052 |  52.7437 | ADAMW    | 0.327362    |               | final_validation_accuracy |
| ADAGRAD       | ADEMAMIX      |  72.797  |  67.36   | ADAGRAD  | 0.0196399   | *             | final_validation_accuracy |
| ADAGRAD       | SOAP          |  72.797  |  52.7437 | ADAGRAD  | 0.253104    |               | final_validation_accuracy |
| ADEMAMIX      | SOAP          |  67.36   |  52.7437 | ADEMAMIX | 0.387034    |               | final_validation_accuracy |

### Validation F1 Score Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.7277 | 0.0039 | 0.0017 | 0.7229 | 0.7325 |
| MILO_LW | 0.7181 | 0.0102 | 0.0046 | 0.7054 | 0.7308 |
| SGD | 0.6881 | 0.0059 | 0.0027 | 0.6807 | 0.6954 |
| ADAMW | 0.6913 | 0.0110 | 0.0049 | 0.6777 | 0.7050 |
| ADAGRAD | 0.7274 | 0.0093 | 0.0041 | 0.7159 | 0.7389 |
| ADEMAMIX | 0.6725 | 0.0349 | 0.0156 | 0.6292 | 0.7158 |
| SOAP | 0.4994 | 0.3719 | 0.1663 | 0.0377 | 0.9612 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric                    |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:--------------------------|
| MILO          | MILO_LW       | 0.727682 | 0.718105 | MILO     | 0.105481    |               | final_validation_f1_score |
| MILO          | SGD           | 0.727682 | 0.688059 | MILO     | 5.46503e-06 | ***           | final_validation_f1_score |
| MILO          | ADAMW         | 0.727682 | 0.691339 | MILO     | 0.000944377 | ***           | final_validation_f1_score |
| MILO          | ADAGRAD       | 0.727682 | 0.727361 | MILO     | 0.945757    |               | final_validation_f1_score |
| MILO          | ADEMAMIX      | 0.727682 | 0.672524 | MILO     | 0.0235951   | *             | final_validation_f1_score |
| MILO          | SOAP          | 0.727682 | 0.499449 | MILO     | 0.241911    |               | final_validation_f1_score |
| MILO_LW       | SGD           | 0.718105 | 0.688059 | MILO_LW  | 0.000998139 | ***           | final_validation_f1_score |
| MILO_LW       | ADAMW         | 0.718105 | 0.691339 | MILO_LW  | 0.00403422  | **            | final_validation_f1_score |
| MILO_LW       | ADAGRAD       | 0.718105 | 0.727361 | ADAGRAD  | 0.171876    |               | final_validation_f1_score |
| MILO_LW       | ADEMAMIX      | 0.718105 | 0.672524 | MILO_LW  | 0.0406598   | *             | final_validation_f1_score |
| MILO_LW       | SOAP          | 0.718105 | 0.499449 | MILO_LW  | 0.25898     |               | final_validation_f1_score |
| SGD           | ADAMW         | 0.688059 | 0.691339 | ADAMW    | 0.577851    |               | final_validation_f1_score |
| SGD           | ADAGRAD       | 0.688059 | 0.727361 | ADAGRAD  | 0.000106975 | ***           | final_validation_f1_score |
| SGD           | ADEMAMIX      | 0.688059 | 0.672524 | SGD      | 0.378869    |               | final_validation_f1_score |
| SGD           | SOAP          | 0.688059 | 0.499449 | SGD      | 0.320163    |               | final_validation_f1_score |
| ADAMW         | ADAGRAD       | 0.691339 | 0.727361 | ADAGRAD  | 0.000560701 | ***           | final_validation_f1_score |
| ADAMW         | ADEMAMIX      | 0.691339 | 0.672524 | ADAMW    | 0.304105    |               | final_validation_f1_score |
| ADAMW         | SOAP          | 0.691339 | 0.499449 | ADAMW    | 0.31293     |               | final_validation_f1_score |
| ADAGRAD       | ADEMAMIX      | 0.727361 | 0.672524 | ADAGRAD  | 0.0222559   | *             | final_validation_f1_score |
| ADAGRAD       | SOAP          | 0.727361 | 0.499449 | ADAGRAD  | 0.242495    |               | final_validation_f1_score |
| ADEMAMIX      | SOAP          | 0.672524 | 0.499449 | ADEMAMIX | 0.357749    |               | final_validation_f1_score |

### Validation Auc Statistics

Number of runs: 5

**Final Epoch Statistics:**

| Optimizer | Mean | Std Dev | Std Error | 95% CI Lower | 95% CI Upper |
|-----------|------|---------|-----------|--------------|--------------|
| MILO | 0.9610 | 0.0010 | 0.0004 | 0.9597 | 0.9622 |
| MILO_LW | 0.9592 | 0.0019 | 0.0008 | 0.9568 | 0.9615 |
| SGD | 0.9565 | 0.0018 | 0.0008 | 0.9542 | 0.9587 |
| ADAMW | 0.9544 | 0.0028 | 0.0013 | 0.9509 | 0.9579 |
| ADAGRAD | 0.9625 | 0.0011 | 0.0005 | 0.9611 | 0.9639 |
| ADEMAMIX | 0.9476 | 0.0086 | 0.0039 | 0.9369 | 0.9583 |
| SOAP | 0.8333 | 0.2053 | 0.0918 | 0.5783 | 1.0882 |

#### Pairwise Significance Tests (Final Epoch)

| Optimizer A   | Optimizer B   |   Mean A |   Mean B | Better   |     p-value | Significant   | Metric               |
|:--------------|:--------------|---------:|---------:|:---------|------------:|:--------------|:---------------------|
| MILO          | MILO_LW       | 0.960957 | 0.959162 | MILO     | 0.105039    |               | final_validation_auc |
| MILO          | SGD           | 0.960957 | 0.956452 | MILO     | 0.00263488  | **            | final_validation_auc |
| MILO          | ADAMW         | 0.960957 | 0.954435 | MILO     | 0.00462337  | **            | final_validation_auc |
| MILO          | ADAGRAD       | 0.960957 | 0.962503 | ADAGRAD  | 0.0485033   | *             | final_validation_auc |
| MILO          | ADEMAMIX      | 0.960957 | 0.947575 | MILO     | 0.0250202   | *             | final_validation_auc |
| MILO          | SOAP          | 0.960957 | 0.833267 | MILO     | 0.236758    |               | final_validation_auc |
| MILO_LW       | SGD           | 0.959162 | 0.956452 | MILO_LW  | 0.0485821   | *             | final_validation_auc |
| MILO_LW       | ADAMW         | 0.959162 | 0.954435 | MILO_LW  | 0.0167867   | *             | final_validation_auc |
| MILO_LW       | ADAGRAD       | 0.959162 | 0.962503 | ADAGRAD  | 0.0120961   | *             | final_validation_auc |
| MILO_LW       | ADEMAMIX      | 0.959162 | 0.947575 | MILO_LW  | 0.0380312   | *             | final_validation_auc |
| MILO_LW       | SOAP          | 0.959162 | 0.833267 | MILO_LW  | 0.242293    |               | final_validation_auc |
| SGD           | ADAMW         | 0.956452 | 0.954435 | SGD      | 0.221669    |               | final_validation_auc |
| SGD           | ADAGRAD       | 0.956452 | 0.962503 | ADAGRAD  | 0.000489422 | ***           | final_validation_auc |
| SGD           | ADEMAMIX      | 0.956452 | 0.947575 | SGD      | 0.0818374   |               | final_validation_auc |
| SGD           | SOAP          | 0.956452 | 0.833267 | SGD      | 0.250892    |               | final_validation_auc |
| ADAMW         | ADAGRAD       | 0.954435 | 0.962503 | ADAGRAD  | 0.00163933  | **            | final_validation_auc |
| ADAMW         | ADEMAMIX      | 0.954435 | 0.947575 | ADAMW    | 0.153307    |               | final_validation_auc |
| ADAMW         | SOAP          | 0.954435 | 0.833267 | ADAMW    | 0.257489    |               | final_validation_auc |
| ADAGRAD       | ADEMAMIX      | 0.962503 | 0.947575 | ADAGRAD  | 0.0173413   | *             | final_validation_auc |
| ADAGRAD       | SOAP          | 0.962503 | 0.833267 | ADAGRAD  | 0.232092    |               | final_validation_auc |
| ADEMAMIX      | SOAP          | 0.947575 | 0.833267 | ADEMAMIX | 0.281316    |               | final_validation_auc |

