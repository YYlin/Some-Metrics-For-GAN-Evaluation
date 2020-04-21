# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 19:40
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : main.py
import argparse
from Inception_score import inception_score
from Kid_score import Kid_score
from Fid_score import Fid_score
import tensorflow as tf


# parsing and configuration
def parse_args():
    desc = "Tensorflow implementation of StarGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='Kid', help='Fid Is or Kid ?')

    parser.add_argument('--real_img', type=str, default='real', help='the path of real image')
    parser.add_argument('--fake_img', type=str, default='fake', help='the path of fake image')
    parser.add_argument('--target_img', type=str, default='target', help='the path of real target image')

    parser.add_argument('--batch_size', type=int, default=15, help='The size of batch size')
    parser.add_argument('--img_size', type=int, default=299, help='The size of image')

    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    sess = tf.InteractiveSession()
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # chose model
    if args.model == 'Is':
        socre = inception_score(sess, args)
    elif args.model == 'Fid':
        socre = Fid_score(sess, args)
    elif args.model == 'Kid':
        socre = Kid_score(sess, args)
    else:
        print('not support %s' % (args.model))
        exit()

    # define placeholders
    print('begin build model')
    socre.build_model()

    # feed data into placeholders
    print('begin training')
    socre.train()


if __name__ == '__main__':
    main()
