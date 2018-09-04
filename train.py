import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import model_input


def train():
    dataset = tf.data.TFRecordDataset(['./data/train_img.tfrecord'])
    dataset = dataset.map(lambda example: model_input.parse_and_preprocess_data(example, 224, 224, True))
    # dataset = dataset.shuffle(20000).batch(FLAGS.batch_size).repeat()
    dataset = dataset.batch(FLAGS.batch_size).repeat()
    iterator = dataset.make_one_shot_iterator()

    images, labels = iterator.get_next()
    logits = model_input.inference(images, 2, True)
    loss = model_input.loss(logits, labels)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = slim.learning.create_train_op(loss, optimizer, summarize_gradients=True)
    variables_to_restore = slim.get_variables_to_restore()
    init_fn = slim.assign_from_checkpoint_fn(FLAGS.vgg16_model_path, variables_to_restore, ignore_missing_vars=True)
    slim.learning.train(train_op=train_op, logdir=FLAGS.log_dir, init_fn=init_fn, save_summaries_secs=20,
                        save_interval_secs=600)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch',
                        default=32)
    parser.add_argument('--log_dir', type=str, help='Directory where to write event logs and checkpoint',
                        default='./log')
    parser.add_argument('--vgg16_model_path', type=str, help='the model ckpt of vgg16',
                        default='./model/vgg_16.ckpt')
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_arguments()
    train()
