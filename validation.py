import tensorflow as tf
import tensorflow.contrib.slim as slim
import model_input
import math
import argparse
import os

tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def validation(model_path, image_size, batch_size, log_dir):
    images, labels = model_input.input_fn(['./data/validation_img.tfrecord'],
                                          batch_size, model_path, image_size, False)
    logits = model_input.inference(model_path, images, 2, False)
    prediction = tf.argmax(logits, axis=1)

    # Choose the metrics to compute:
    value_op, update_op = tf.metrics.accuracy(labels, prediction)
    num_batchs = math.ceil(model_input.VALIDATION_EXAMPLES_NUM / batch_size)

    print('Running evaluation...')
    # Only load latest checkpoint
    checkpoint_path = tf.train.latest_checkpoint(log_dir)

    metric_values = slim.evaluation.evaluate_once(
        num_evals=num_batchs,
        master='',
        checkpoint_path=checkpoint_path,
        logdir=log_dir,
        eval_op=update_op,
        final_op=value_op)
    print('model: {}, acc: {}'.format(checkpoint_path, metric_values))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch',
                        default=32)
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
    parser.add_argument('--resnet_v1_50_model_path', type=str, help='the model ckpt of resnet_v1_50',
                        default='./model/resnet_v1_50.ckpt')
    parser.add_argument('--resnet_v1_50_image_size', type=int, help='the size of input image of model resnet_v1_50',
                        default=224)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_arguments()
    # validation(FLAGS.vgg16_model_path, FLAGS.vgg16_image_size, FLAGS.batch_size, FLAGS.log_dir)
    # validation(FLAGS.inception_v3_model_path, FLAGS.inception_v3_image_size, FLAGS.batch_size, FLAGS.log_dir)
    validation(FLAGS.resnet_v1_50_model_path, FLAGS.resnet_v1_50_image_size, FLAGS.batch_size, FLAGS.log_dir)
