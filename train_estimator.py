import tensorflow as tf
import os
import argparse
import model_input

tf.logging.set_verbosity(tf.logging.INFO)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train(model_path, image_size):
    my_checkpoint_config = tf.estimator.RunConfig(save_checkpoints_steps=100, keep_checkpoint_max=5)

    mnist_classifier = tf.estimator.Estimator(model_fn=model_input.model_fn,
                                              model_dir=FLAGS.log_dir,
                                              config=my_checkpoint_config,
                                              params={'class_num': 2, 'model_path': model_path})
    tensor_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=100)

    mnist_classifier.train(
        input_fn=lambda: model_input.input_fn(['./data/train_img.tfrecord'],
                                              FLAGS.batch_size, model_path, image_size, True),
        steps=FLAGS.max_step)

    # eval_results = mnist_classifier.evaluate(
    #     input_fn=lambda: model_input.input_fn(['.data/validation_img.tfrecord'],
    #                                           FLAGS.batch_size, model_path, image_size, False))
    # print('validation acc: {}'.format(eval_results))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch',
                        default=32)
    parser.add_argument('--max_step', type=int, help='Number of steps to run trainer',
                        default=200)
    parser.add_argument('--log_dir', type=str, help='Directory where to write event logs and checkpoint',
                        default='./log')
    parser.add_argument('--vgg16_model_path', type=str, help='the model ckpt of vgg16',
                        default='./model/vgg_16.ckpt')
    parser.add_argument('--vgg16_image_size', type=int, help='the size of input image of model vgg16',
                        default=224)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_arguments()
    train(model_path=FLAGS.vgg16_model_path, image_size=FLAGS.vgg16_image_size)
