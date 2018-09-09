import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import model_input
import os
import math
import numpy as np
from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)


def train_slim(model_path, image_size):
    images, labels = model_input.input_fn(['./data/train_img.tfrecord'], FLAGS.batch_size, model_path, image_size, True)
    logits = model_input.inference(model_path, images, 2, True)
    loss = model_input.loss(logits, labels)

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = slim.learning.create_train_op(loss, optimizer, summarize_gradients=True)
    variables_to_restore = slim.get_variables_to_restore()
    init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore, ignore_missing_vars=True)
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


def train(model_path, image_size):
    training_dataset = tf.data.TFRecordDataset(['./data/train_img.tfrecord'])
    training_dataset = training_dataset.map(
        lambda example: model_input.parse_and_preprocess_data(example, model_path, image_size, True))
    # dataset = dataset.shuffle(20000).batch(FLAGS.batch_size).repeat()
    training_dataset = training_dataset.batch(FLAGS.batch_size).repeat()

    validation_dataset = tf.data.TFRecordDataset(['./data/validation_img.tfrecord'])
    validation_dataset = validation_dataset.map(
        lambda example: model_input.parse_and_preprocess_data(example, model_path, image_size, False))
    validation_dataset = validation_dataset.batch(FLAGS.batch_size)

    iterator = tf.data.Iterator.from_structure(output_types=training_dataset.output_types,
                                               output_shapes=training_dataset.output_shapes)

    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    images, labels = iterator.get_next()
    is_training = tf.placeholder(dtype=tf.bool)
    logits = model_input.inference(model_path, images, 2, is_training)
    pred = tf.nn.softmax(logits)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    loss = model_input.loss(logits, labels)

    variables_to_train, variables_to_restore = model_input.variables_to_restore_and_train(model_path)
    global_step = tf.train.get_or_create_global_step()
    train_op = model_input.get_train_op(loss, variables_to_train, variables_to_restore, FLAGS.batch_size,
                                        FLAGS.learning_rate, global_step)

    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # 先初始化所有变量，避免有些变量未读取而产生错误
        init = tf.global_variables_initializer()
        sess.run(init)

        # 建立一个从预训练模型checkpoint中读取上述列表中的相应变量的参数的函数
        init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore, ignore_missing_vars=True)
        # restore模型参数
        init_fn(sess)
        saver = tf.train.Saver()
        sess.run(training_init_op)
        print('begin to train!')
        ckpt = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, ckpt, 0)
        train_step = 0
        while train_step < FLAGS.max_step:
            _, train_loss, logits_op, pred_op, labels_op = sess.run([train_op, loss, logits, pred, labels],
                                                                    feed_dict={is_training: True})
            # print('logits: {}, pred:{}, labels:{}, loss: {}'.format(logits_op, pred_op, labels_op, train_loss))
            train_step += 1
            if train_step % 100 == 0:
                saver.save(sess, ckpt, train_step)
                # print('step: {}, loss: {}'.format(train_step, train_loss))
                sess.run(validation_init_op)
                precision = evaluate(sess, top_k_op, is_training, model_input.VALIDATION_EXAMPLES_NUM)
                print('step: {}, loss: {}, validation precision: {}'.format(train_step, train_loss, precision))
                sess.run(training_init_op)
            if train_step == FLAGS.max_step and train_step % 100 != 0:
                saver.save(sess, ckpt, train_step)
                print('step: {}, loss: {}'.format(train_step, train_loss))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate.',
                        default=0.001)
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch',
                        default=32)
    parser.add_argument('--max_step', type=int, help='Number of steps to run trainer',
                        default=2000)
    parser.add_argument('--log_dir', type=str, help='Directory where to write event logs and checkpoint',
                        default='./log')
    parser.add_argument('--vgg16_model_path', type=str, help='the model ckpt of vgg16',
                        default='./model/vgg_16.ckpt')
    parser.add_argument('--vgg16_image_size', type=int, help='the size of input image of model vgg16',
                        default=224)
    parser.add_argument('--inception_v3_model_path', type=str, help='the model ckpt of inception_v3',
                        default='./model/inception_v3.ckpt')
    parser.add_argument('--inception_v3_image_size', type=int, help='the size of input image of model inception_v3',
                        default=299)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_arguments()
    # train(model_path=FLAGS.vgg16_model_path, image_size=FLAGS.vgg16_image_size)
    # train_slim(model_path=FLAGS.vgg16_model_path, image_size=FLAGS.vgg16_image_size)
    # train_slim(model_path=FLAGS.inception_v3_model_path, image_size=FLAGS.inception_v3_image_size)
    train(model_path=FLAGS.inception_v3_model_path, image_size=FLAGS.inception_v3_image_size)

