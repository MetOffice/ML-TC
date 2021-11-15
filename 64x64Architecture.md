Generator:
| Layer                            |  Output Size  |
|----------------------------------|:-------------:|
| Input                            |  100 x 1 x 1  |
| TransConv, BatchNorm, Leaky Relu |  512 x 4 x 4  |
| TransConv, BatchNorm, Leaky Relu |  256 x 8 x 8  |
| TransConv, BatchNorm, Leaky Relu | 128 x 16 x 16 |
| TransConv, BatchNorm, Leaky Relu |  64 x 32 x 32 |
| TransConv, Leaky Relu            |  1 x 64 x 64  |

Discriminator:
| Layer                       |  Output Size  |
|-----------------------------|:-------------:|
| Input                       |  1 x 64 x 64  |
| Conv, Leaky Relu            |  64 x 32 x 32 |
| Conv, BatchNorm, Leaky Relu | 128 x 16 x 16 |
| Conv, BatchNorm, Leaky Relu |  256 x 8 x 8  |
| Conv, BatchNorm, Leaky Relu |  512 x 4 x 4  |
| Conv, Sigmoid               |       1       |
