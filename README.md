# the three metrics for GAN evaluation
Collection of metrics for GAN evaluation, e.g. 'Fréchet Inception Distance'(FID), 'Kernel Inception Distance', 'Inception score' (IS) in Tensorflow.

## Some configurations for our model
sudo pip install tensorflow-gan==1.0.0.dev0
sudo pip install tensorflow-gpu==1.15.0

## Lists
| name     | paper     |
| :-----------:  | :-----------: |
| Fréchet Inception Distance |  [KID](https://arxiv.org/abs/1706.08500)|
| :-----------:  | :-----------: |
| Kernel Inception Distance | [KID](https://arxiv.org/abs/1801.01401)|
| :-----------:  | :-----------: |
| Inception score | [IS](https://arxiv.org/abs/1606.03498)|

## Test
By using different commands, you canYou can chose different metrics：

| metrics | commands     |
| :-----------:  | :-----------: |
| Fréchet Inception Distance | 'python main.py --model Fid'|
| :-----------:  | :-----------: |
| Kernel Inception Distance | 'python main.py --model Kid'|
| :-----------:  | :-----------: |
| Inception score | 'python main.py --model Is'|
