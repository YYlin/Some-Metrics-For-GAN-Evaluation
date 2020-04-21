# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 21:20
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : Fid_score.py
from util import *


class Fid_score(object):
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

    def _symmetric_matrix_square_root(self, mat, eps=1e-10):
        """Compute square root of a symmetric matrix.
        Note that this is different from an elementwise square root. We want to
        compute M' where M' = sqrt(mat) such that M' * M' = mat.
        Also note that this method **only** works for symmetric matrices.
        Args:
          mat: Matrix to take the square root of.
          eps: Small epsilon such that any element less than eps will not be square
            rooted to guard against numerical instability.
        Returns:
          Matrix square root of mat.
        """
        # Unlike numpy, tensorflow's return order is (s, u, v)
        s, u, v = linalg_ops.svd(mat)
        # sqrt is unstable around 0, just use 0 in such case
        si = array_ops.where(math_ops.less(s, eps), s, math_ops.sqrt(s))
        # Note that the v returned by Tensorflow is v = V
        # (when referencing the equation A = U S V^T)
        # This is unlike Numpy which returns v = V^T
        return math_ops.matmul(
            math_ops.matmul(u, array_ops.diag(si)), v, transpose_b=True)


    def trace_sqrt_product(self, sigma, sigma_v):
        """Find the trace of the positive sqrt of product of covariance matrices.
        '_symmetric_matrix_square_root' only works for symmetric matrices, so we
        cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
        ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
        Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
        We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
        Note the following properties:
        (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
           => eigenvalues(A A B B) = eigenvalues (A B B A)
        (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
           => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
        (iii) forall M: trace(M) = sum(eigenvalues(M))
           => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                         = sum(sqrt(eigenvalues(A B B A)))
                                         = sum(eigenvalues(sqrt(A B B A)))
                                         = trace(sqrt(A B B A))
                                         = trace(sqrt(A sigma_v A))
        A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
        use the _symmetric_matrix_square_root function to find the roots of these
        matrices.
        Args:
          sigma: a square, symmetric, real, positive semi-definite covariance matrix
          sigma_v: same as sigma
        Returns:
          The trace of the positive square root of sigma*sigma_v
        """

        # Note sqrt_sigma is called "A" in the proof above
        sqrt_sigma = self._symmetric_matrix_square_root(sigma)

        # This is sqrt(A sigma_v A) above
        sqrt_a_sigmav_a = math_ops.matmul(sqrt_sigma,
                                          math_ops.matmul(sigma_v, sqrt_sigma))

        return math_ops.trace(self._symmetric_matrix_square_root(sqrt_a_sigmav_a))


    # refer https://github.com/taki0112/GAN_Metrics-Tensorflow/blob/master/frechet_kernel_Inception_distance.py
    def frechet_classifier_distance_from_activations(self, real_activations,  generated_activations):

        real_activations.shape.assert_has_rank(2)
        generated_activations.shape.assert_has_rank(2)

        activations_dtype = real_activations.dtype
        if activations_dtype != dtypes.float64:
            real_activations = math_ops.to_double(real_activations)
            generated_activations = math_ops.to_double(generated_activations)

        # Compute mean and covariance matrices of activations.
        m = math_ops.reduce_mean(real_activations, 0)
        m_w = math_ops.reduce_mean(generated_activations, 0)
        num_examples_real = math_ops.to_double(array_ops.shape(real_activations)[0])
        num_examples_generated = math_ops.to_double(
            array_ops.shape(generated_activations)[0])

        # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
        real_centered = real_activations - m
        sigma = math_ops.matmul(
            real_centered, real_centered, transpose_a=True) / (
                        num_examples_real - 1)

        gen_centered = generated_activations - m_w
        sigma_w = math_ops.matmul(
            gen_centered, gen_centered, transpose_a=True) / (
                          num_examples_generated - 1)

        # Find the Tr(sqrt(sigma sigma_w)) component of FID
        sqrt_trace_component = self.trace_sqrt_product(sigma, sigma_w)

        # Compute the two components of FID.

        # First the covariance component.
        # Here, note that trace(A + B) = trace(A) + trace(B)
        trace = math_ops.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

        # Next the distance between means.
        mean = math_ops.reduce_sum(
            math_ops.squared_difference(m, m_w))  # Equivalent to L2 but more stable.
        fid = trace + mean
        if activations_dtype != dtypes.float64:
            fid = math_ops.cast(fid, activations_dtype)

        return fid

    # the output of IS is class, however Fid is pool_3
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
        self.fcd = self.frechet_classifier_distance_from_activations(self.real_activation, self.fake_activation)


    def train(self):
        batches = len(self.real) // self.batch_size

        print('calculate the feature of real image')
        real_fea = np.zeros([batches * self.batch_size, 2048], dtype=np.float32)
        for i in range(batches):
            real = self.real[i * self.batch_size:(i + 1) * self.batch_size]
            real_fea[i * self.batch_size:(i + 1) * self.batch_size] = self.sess.run(self.fea, feed_dict={self.inception_images: real})

        print('calculate the feature of fake image')
        fake_fea = np.zeros([batches * self.batch_size, 2048], dtype=np.float32)
        for i in range(batches):
            fake = self.fake[i * self.batch_size:(i + 1) * self.batch_size]
            fake_fea[i * self.batch_size:(i + 1) * self.batch_size] = self.sess.run(self.fea, feed_dict={self.inception_images: fake})

        print('calculate fid between real image and fake image')
        Fid = self.sess.run(self.fcd, feed_dict={self.real_activation: real_fea, self.fake_activation: fake_fea})
        print('The Fid Score is %d:'%Fid)

