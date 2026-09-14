| model               | condition              |   coverage |   selective_error |   accuracy |   unsupported_release_rate |
|:--------------------|:-----------------------|-----------:|------------------:|-----------:|---------------------------:|
| logistic_regression | clean_test             |     0.7070 |            0.0038 |     0.9728 |                     0.0026 |
| logistic_regression | gaussian_noise_0.1     |     0.7105 |            0.0038 |     0.9719 |                     0.0026 |
| logistic_regression | gaussian_noise_0.2     |     0.7061 |            0.0038 |     0.9746 |                     0.0026 |
| logistic_regression | random_missingness_10% |     0.6561 |            0.0041 |     0.9719 |                     0.0026 |
| logistic_regression | random_missingness_20% |     0.5974 |            0.0032 |     0.9675 |                     0.0018 |
| random_forest       | clean_test             |     0.7246 |            0.0036 |     0.9500 |                     0.0026 |
| random_forest       | gaussian_noise_0.1     |     0.7035 |            0.0038 |     0.9491 |                     0.0026 |
| random_forest       | gaussian_noise_0.2     |     0.6325 |            0.0042 |     0.9509 |                     0.0026 |
| random_forest       | random_missingness_10% |     0.5886 |            0.0028 |     0.9482 |                     0.0018 |
| random_forest       | random_missingness_20% |     0.4474 |            0.0100 |     0.9447 |                     0.0044 |