# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 21:53
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : Kid_score.py
import tensorflow as tf
from util import *
from tensorflow.python.ops import control_flow_ops


class Kid_score(object):
    def __init__(self, sess, args):
        self.fake_img = args.fake_img
        self.real_img = args.real_img
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.sess = sess

        self.fake = get_images(self.fake_img, self.img_size)
        self.real = get_images(self.real_img, self.img_size)

        if len(self.fake) != len(self.real):
            print('The length of generated image must be equal to real image')
            exit()
        else:
            print('The length of real image is %d'%(len(self.real)))

    # refer https://github.com/taki0112/GAN_Metrics-Tensorflow/blob/master/frechet_kernel_Inception_distance.py
    def kernel_classifier_distance_and_std_from_activations(self, real_activations,
                                                            generated_activations,
                                                            max_block_size=10,
                                                            dtype=None):
        """Kernel "classifier" distance for evaluating a generative model.
        This methods computes the kernel classifier distance from activations of
        real images and generated images. This can be used independently of the
        kernel_classifier_distance() method, especially in the case of using large
        batches during evaluation where we would like to precompute all of the
        activations before computing the classifier distance, or if we want to
        compute multiple metrics based on the same images. It also returns a rough
        estimate of the standard error of the estimator.
        This technique is described in detail in https://arxiv.org/abs/1801.01401.
        Given two distributions P and Q of activations, this function calculates
            E_{X, X' ~ P}[k(X, X')] + E_{Y, Y' ~ Q}[k(Y, Y')]
              - 2 E_{X ~ P, Y ~ Q}[k(X, Y)]
        where k is the polynomial kernel
            k(x, y) = ( x^T y / dimension + 1 )^3.
        This captures how different the distributions of real and generated images'
        visual features are. Like the Frechet distance (and unlike the Inception
        score), this is a true distance and incorporates information about the
        target images. Unlike the Frechet score, this function computes an
        *unbiased* and asymptotically normal estimator, which makes comparing
        estimates across models much more intuitive.
        The estimator used takes time quadratic in max_block_size. Larger values of
        max_block_size will decrease the variance of the estimator but increase the
        computational cost. This differs slightly from the estimator used by the
        original paper; it is the block estimator of https://arxiv.org/abs/1307.1954.
        The estimate of the standard error will also be more reliable when there are
        more blocks, i.e. when max_block_size is smaller.
        NOTE: the blocking code assumes that real_activations and
        generated_activations are both in random order. If either is sorted in a
        meaningful order, the estimator will behave poorly.
        Args:
          real_activations: 2D Tensor containing activations of real data. Shape is
            [batch_size, activation_size].
          generated_activations: 2D Tensor containing activations of generated data.
            Shape is [batch_size, activation_size].
          max_block_size: integer, default 1024. The distance estimator splits samples
            into blocks for computational efficiency. Larger values are more
            computationally expensive but decrease the variance of the distance
            estimate. Having a smaller block size also gives a better estimate of the
            standard error.
          dtype: if not None, coerce activations to this dtype before computations.
        Returns:
         The Kernel Inception Distance. A floating-point scalar of the same type
           as the output of the activations.
         An estimate of the standard error of the distance estimator (a scalar of
           the same type).
        """

        real_activations.shape.assert_has_rank(2)
        generated_activations.shape.assert_has_rank(2)
        real_activations.shape[1].assert_is_compatible_with(
            generated_activations.shape[1])

        if dtype is None:
            dtype = real_activations.dtype
            assert generated_activations.dtype == dtype
        else:
            real_activations = math_ops.cast(real_activations, dtype)
            generated_activations = math_ops.cast(generated_activations, dtype)

        # Figure out how to split the activations into blocks of approximately
        # equal size, with none larger than max_block_size.
        n_r = array_ops.shape(real_activations)[0]
        n_g = array_ops.shape(generated_activations)[0]

        n_bigger = math_ops.maximum(n_r, n_g)
        n_blocks = math_ops.to_int32(math_ops.ceil(n_bigger / max_block_size))

        v_r = n_r // n_blocks
        v_g = n_g // n_blocks

        n_plusone_r = n_r - v_r * n_blocks
        n_plusone_g = n_g - v_g * n_blocks

        sizes_r = array_ops.concat([
            array_ops.fill([n_blocks - n_plusone_r], v_r),
            array_ops.fill([n_plusone_r], v_r + 1),
        ], 0)
        sizes_g = array_ops.concat([
            array_ops.fill([n_blocks - n_plusone_g], v_g),
            array_ops.fill([n_plusone_g], v_g + 1),
        ], 0)

        zero = array_ops.zeros([1], dtype=dtypes.int32)
        inds_r = array_ops.concat([zero, math_ops.cumsum(sizes_r)], 0)
        inds_g = array_ops.concat([zero, math_ops.cumsum(sizes_g)], 0)

        dim = math_ops.cast(tf.shape(real_activations)[1], dtype)

        def compute_kid_block(i):
            'Compute the ith block of the KID estimate.'
            r_s = inds_r[i]
            r_e = inds_r[i + 1]
            r = real_activations[r_s:r_e]
            m = math_ops.cast(r_e - r_s, dtype)

            g_s = inds_g[i]
            g_e = inds_g[i + 1]
            g = generated_activations[g_s:g_e]
            n = math_ops.cast(g_e - g_s, dtype)

            k_rr = (math_ops.matmul(r, r, transpose_b=True) / dim + 1) ** 3
            k_rg = (math_ops.matmul(r, g, transpose_b=True) / dim + 1) ** 3
            k_gg = (math_ops.matmul(g, g, transpose_b=True) / dim + 1) ** 3
            return (-2 * math_ops.reduce_mean(k_rg) +
                    (math_ops.reduce_sum(k_rr) - math_ops.trace(k_rr)) / (m * (m - 1)) +
                    (math_ops.reduce_sum(k_gg) - math_ops.trace(k_gg)) / (n * (n - 1)))

        ests = tf.map_fn(
            compute_kid_block, math_ops.range(n_blocks), dtype=dtype, back_prop=False)

        mn = math_ops.reduce_mean(ests)

        # nn_impl.moments doesn't use the Bessel correction, which we want here
        n_blocks_ = math_ops.cast(n_blocks, dtype)
        var = control_flow_ops.cond(
            math_ops.less_equal(n_blocks, 1),
            lambda: array_ops.constant(float('nan'), dtype=dtype),
            lambda: math_ops.reduce_sum(math_ops.square(ests - mn)) / (n_blocks_ - 1))

        return mn, math_ops.sqrt(var / n_blocks_)

    def inception_activations(self, images, num_splits=1):
        generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
        activations = tf.map_fn(
            fn=functools.partial(tfgan.eval.run_inception, output_tensor='pool_3:0'),
            elems=array_ops.stack(generated_images_list),
            parallel_iterations=1,
            back_prop=False,
            swap_memory=True,
            name='RunClassifier')
        activations = array_ops.concat(array_ops.unstack(activations), 0)
        return activations

    def build_model(self):

        self.inception_images = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, 3])
        self.fea = self.inception_activations(self.inception_images)

        self.real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
        self.fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')
        self.kid = self.kernel_classifier_distance_and_std_from_activations(self.real_activation, self.fake_activation)

    def train(self):
        n_batches = len(self.real) // self.batch_size
        real_fea = np.zeros([n_batches * self.batch_size, 2048], dtype=np.float32)

        print('calculate the feature of real image')
        for i in range(n_batches):
            real = self.real[i * self.batch_size:(i + 1) * self.batch_size]
            real_fea[i * self.batch_size:(i + 1) * self.batch_size] = self.sess.run(self.fea, feed_dict={self.inception_images: real})

        print('calculate the feature of fake image')
        fake_fea = np.zeros([n_batches * self.batch_size, 2048], dtype=np.float32)
        for i in range(n_batches):
            fake = self.fake[i * self.batch_size:(i + 1) * self.batch_size]
            fake_fea[i * self.batch_size:(i + 1) * self.batch_size] = self.sess.run(self.fea, feed_dict={self.inception_images: fake})

        print('calculate fid between real image and fake image')
        Kid = self.sess.run(self.kid, feed_dict={self.real_activation: real_fea, self.fake_activation: fake_fea})
        print('The Kid is %d'%Kid)


