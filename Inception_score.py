# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 19:42
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : Inception_score.py
from util import *
from tensorflow.python.ops import array_ops
import functools
import time


class inception_score(object):
    def __init__(self, sess, args):
        self.fake_img = args.fake_img
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.sess = sess
        self.data = get_images(self.fake_img, self.img_size)
        print('The length of dataset', len(self.data), self.data.shape)

    # refer https://docs.w3cub.com/tensorflow~python/tf/contrib/gan/eval/run_inception/
    def inception_logits(self, images, num_splits=1):
        generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
        logits = tf.map_fn(
            fn=functools.partial(tfgan.eval.run_inception, output_tensor='logits:0'),
            elems=array_ops.stack(generated_images_list),
            parallel_iterations=1,
            back_prop=False,
            swap_memory=True,
            name='RunClassifier')
        print('logits:', logits)
        logits = array_ops.concat(array_ops.unstack(logits), 0)
        return logits

    def preds2score(self, preds, splits=1):
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)

    def build_model(self):
        self.inception_images = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, 3])
        self.logits = self.inception_logits(self.inception_images)
        print('self.logits:', self.logits)

    def train(self):
        batches = len(self.data) // self.batch_size
        preds = np.zeros([batches * self.batch_size, 1000], dtype=np.float32)

        for i in range(batches):
            print('Begin Calculating Inception Score with %i images ' % (i))
            start_time = time.time()
            img = self.data[i * self.batch_size:(i + 1) * self.batch_size]
            preds[i * self.batch_size:(i + 1) * self.batch_size] = self.sess.run(self.logits, feed_dict={self.inception_images: img})[:, :1000]
            print('Finish Inception Score calculation time: %f s' % (time.time() - start_time))

        preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
        mean, std = self.preds2score(preds)
        print('Inception Score is : %f, %f' % (mean, std))

