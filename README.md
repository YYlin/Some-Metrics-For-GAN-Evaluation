# Three metrics for GAN evaluation
Collection of metrics for GAN evaluation, e.g. 'Fréchet Inception Distance'(FID), 'Kernel Inception Distance', 'Inception score' (IS) in Tensorflow.


## Some configurations for our model
>sudo pip install tensorflow-gan==1.0.0.dev0  
sudo pip install tensorflow-gpu==1.15.0


## Test
| metrics                    | paper                                   | Performance score. | commands                     |
| ----------                 | :-----------:                           | :-----------:      | :-----------:                |
| Fréchet Inception Distance |  [FID](https://arxiv.org/abs/1706.08500)| Higher is better.  | `python main.py --model Fid` |
|                            |                                         |                    |                              |
| Kernel Inception Distance  | [KID](https://arxiv.org/abs/1801.01401) | Lower is better.   | `python main.py --model Kid` |
|                            |                                         |                    |                              |
| Inception score            | [IS](https://arxiv.org/abs/1606.03498)  | Lower is better.   | `python main.py --model Is`  |


## Reference
[GAN_Metrics-Tensorflow](https://github.com/hwalsuklee/tensorflow-generative-model-collections)  

[tf.contrib.gan.eval.run_inception](https://docs.w3cub.com/tensorflow~python/tf/contrib/gan/eval/run_inception/)

