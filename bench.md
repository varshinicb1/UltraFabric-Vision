# UltraFabric-Vision - Inference Benchmark

- Device: **cpu**
- torch 2.11.0+cpu | Fast-mode detector: `vit_ae`

## CPU

| Path | mean | p50 | p95 | min | max | FPS |
|------|-----:|----:|----:|----:|----:|----:|
| **Accurate (ensemble)** |  454.88 |  443.91 |  536.97 |  395.44 |  738.78 |    2.2 |
| **Fast (vit_ae)** |   28.23 |   25.69 |   40.60 |   22.72 |  103.64 |   35.4 |
| _preprocess_ |    0.89 |    0.82 |    1.44 |    0.58 |    1.77 | 1118.3 |
| _patchcore_ |  170.10 |  168.39 |  180.23 |  162.83 |  256.85 |    5.9 |
| _dino_ |  277.32 |  271.42 |  323.68 |  240.46 |  403.78 |    3.6 |
| _vit_ae_ |   25.31 |   24.93 |   29.09 |   22.23 |   38.49 |   39.5 |

## CPU

| Path | mean | p50 | p95 | min | max | FPS |
|------|-----:|----:|----:|----:|----:|----:|
| **Accurate (ensemble)** |  453.28 |  445.98 |  505.60 |  408.66 |  601.48 |    2.2 |
| **Fast (vit_ae)** |   26.99 |   26.56 |   31.55 |   24.09 |   33.89 |   37.1 |
| _preprocess_ |    0.77 |    0.75 |    0.94 |    0.53 |    2.05 | 1302.5 |
| _patchcore_ |  176.64 |  174.98 |  189.61 |  161.13 |  319.65 |    5.7 |
| _dino_ |  267.49 |  264.46 |  294.16 |  225.94 |  426.32 |    3.7 |
| _vit_ae_ |   24.33 |   24.04 |   27.27 |   21.98 |   29.94 |   41.1 |

_Latencies in ms; each includes preprocessing for end-to-end rows and excludes it for component rows. Measured with warm-up and CUDA sync._