import tensorflow as tf
import tensorflow.contrib.slim as slim
import model_input
import math
import argparse


def validation():
    dataset = tf.data.TFRecordDataset(['./data/validation_img.tfrecord'])
    dataset = dataset.map(lambda example: model_input.parse_and_preprocess_data(example, 224, 224, False))
    dataset = dataset.batch(FLAGS.batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    logits = model_input.inference(images, 2, False)
    prediction = tf.argmax(tf.nn.softmax(logits), axis=1)

    # Choose the metrics to compute:
    value_op, update_op = tf.metrics.accuracy(labels, prediction)
    num_batchs = math.ceil(5000 / FLAGS.batch_size)

    print('Running evaluation...')
    # Only load latest checkpoint
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.log_dir)

    metric_values = slim.evaluation.evaluate_once(
        num_evals=num_batchs,
        master='',
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.log_dir,
        eval_op=update_op,
        final_op=value_op)
    print('model: {}, acc: {}'.format(checkpoint_path, metric_values))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch',
                        default=32)
    parser.add_argument('--log_dir', type=str, help='Directory where to write event logs and checkpoint',
                        default='./log')
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_arguments()
    validation()