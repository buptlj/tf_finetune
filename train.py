import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import model_input
import os
import math
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


def train_slim():
    images, labels = model_input.input_fn(['./data/train_img.tfrecord'], FLAGS.batch_size, 224, 224, True)
    logits = model_input.inference(images, 2, True)
    loss = model_input.loss(logits, labels)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = slim.learning.create_train_op(loss, optimizer, summarize_gradients=True)
    variables_to_restore = slim.get_variables_to_restore()
    init_fn = slim.assign_from_checkpoint_fn(FLAGS.vgg16_model_path, variables_to_restore, ignore_missing_vars=True)
    slim.learning.train(train_op=train_op, logdir=FLAGS.log_dir,
                        log_every_n_steps=100, number_of_steps=FLAGS.max_step,
                        init_fn=init_fn, save_summaries_secs=120,
                        save_interval_secs=600)


def evaluate(sess, top_k_op, training, examples):
    iter_per_epoch = int(math.ceil(examples / FLAGS.batch_size))
    # total_sample = iter_per_epoch * FLAGS.batch_size
    correct_predict = 0
    step = 0

    while step < iter_per_epoch:
        predict = sess.run(top_k_op, feed_dict={training: False})
        correct_predict += np.sum(predict)
        step += 1

    precision = correct_predict / examples
    return precision


def train():
    training_dataset = tf.data.TFRecordDataset(['./data/train_img.tfrecord'])
    training_dataset = training_dataset.map(lambda example:
                                            model_input.parse_and_preprocess_data(example, 224, 224, True))
    # dataset = dataset.shuffle(20000).batch(FLAGS.batch_size).repeat()
    training_dataset = training_dataset.batch(FLAGS.batch_size).repeat()

    validation_dataset = tf.data.TFRecordDataset(['./data/validation_img.tfrecord'])
    validation_dataset = validation_dataset.map(lambda example:
                                                model_input.parse_and_preprocess_data(example, 224, 224, False))
    validation_dataset = validation_dataset.batch(FLAGS.batch_size)

    iterator = tf.data.Iterator.from_structure(output_types=training_dataset.output_types,
                                               output_shapes=training_dataset.output_shapes)

    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    images, labels = iterator.get_next()
    is_training = tf.placeholder(dtype=tf.bool)
    logits = model_input.inference(images, 2, is_training)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    loss = model_input.loss(logits, labels)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        # 先初始化所有变量，避免有些变量未读取而产生错误
        init = tf.global_variables_initializer()
        sess.run(init)
        # exclusions = ['vgg_16/dropout7', 'vgg_16/fc8']
        # 创建一个列表，包含除了exclusions之外所有需要读取的变量
        variables_to_restore = slim.get_variables_to_restore()
        # 建立一个从预训练模型checkpoint中读取上述列表中的相应变量的参数的函数
        init_fn = slim.assign_from_checkpoint_fn(FLAGS.vgg16_model_path, variables_to_restore,
                                                 ignore_missing_vars=True)
        # restore模型参数
        init_fn(sess)
        saver = tf.train.Saver()
        sess.run(training_init_op)
        print('begin to train!')
        ckpt = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, ckpt, 0)
        train_step = 0
        while train_step < FLAGS.max_step:
            _, train_loss = sess.run([train_op, loss], feed_dict={is_training: True})
            train_step += 1
            if train_step % 100 == 0:
                saver.save(sess, ckpt, train_step)
                # print('step: {}, loss: {}'.format(train_step, train_loss))
                sess.run(validation_init_op)
                precision = evaluate(sess, top_k_op, is_training, model_input.VALIDATION_EXAMPLES_NUM)
                print('step: {}, loss: {}, validation precision: {}'.format(train_step, train_loss, precision))
                sess.run(training_init_op)
            if train_step == FLAGS.max_step:
                saver.save(sess, ckpt, train_step)
                print('step: {}, loss: {}'.format(train_step, train_loss))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch',
                        default=32)
    parser.add_argument('--max_step', type=int, help='Number of steps to run trainer',
                        default=1250)
    parser.add_argument('--log_dir', type=str, help='Directory where to write event logs and checkpoint',
                        default='./log')
    parser.add_argument('--vgg16_model_path', type=str, help='the model ckpt of vgg16',
                        default='./model/vgg_16.ckpt')
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_arguments()
    # train()
    train_slim()

