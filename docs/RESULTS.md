## Defocus Blur sev 1 - Chi2

| Comparison                      |   Chi2_Stat |     p-value |   Shared_Fragile_Count | Significant   |
|:--------------------------------|------------:|------------:|-----------------------:|:--------------|
| ResNet-50 - ResNet-152          |    151.634  | 7.61627e-35 |                     12 | True          |
| ResNet-50 - ConvNeXt-Base       |    206.543  | 7.79958e-47 |                     14 | True          |
| ResNet-50 - ViT-B/16            |    158.044  | 3.0264e-36  |                     14 | True          |
| ResNet-50 - EfficientNet-B4     |     64.0929 | 1.18691e-15 |                      7 | True          |
| ResNet-152 - ConvNeXt-Base      |     56.0592 | 7.0321e-14  |                      9 | True          |
| ResNet-152 - ViT-B/16           |     82.7571 | 9.27821e-20 |                     12 | True          |
| ResNet-152 - EfficientNet-B4    |     65.7294 | 5.17292e-16 |                      8 | True          |
| ConvNeXt-Base - ViT-B/16        |    155.148  | 1.3e-35     |                     16 | True          |
| ConvNeXt-Base - EfficientNet-B4 |    192.242  | 1.03051e-43 |                     13 | True          |
| ViT-B/16 - EfficientNet-B4      |     80.5505 | 2.83374e-19 |                     10 | True          |


## Defocus Blur sev 1 - Fisher

| Comparison                      |   Odds_Ratio |    p-value |   Shared_Fragile | Significant   |
|:--------------------------------|-------------:|-----------:|-----------------:|:--------------|
| ResNet-50 - ResNet-152          |      44.0769 | 4.697e-13  |               12 | True          |
| ResNet-50 - ConvNeXt-Base       |      64.0383 | 2.8529e-16 |               14 | True          |
| ResNet-50 - ViT-B/16            |      43.0455 | 1.6728e-14 |               14 | True          |
| ResNet-50 - EfficientNet-B4     |      23.309  | 4.6909e-07 |                7 | True          |
| ResNet-152 - ConvNeXt-Base      |      15.3913 | 2.4421e-07 |                9 | True          |
| ResNet-152 - ViT-B/16           |      18.76   | 7.2068e-10 |               12 | True          |
| ResNet-152 - EfficientNet-B4    |      21.1778 | 1.5413e-07 |                8 | True          |
| ConvNeXt-Base - ViT-B/16        |      34.0633 | 2.9799e-15 |               16 | True          |
| ConvNeXt-Base - EfficientNet-B4 |      62.205  | 3.6493e-15 |               13 | True          |
| ViT-B/16 - EfficientNet-B4      |      22.7163 | 4.33e-09   |               10 | True          |