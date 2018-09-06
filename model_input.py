import tensorflow as tf
import numpy as np
import os
import cv2
from slim_module.nets import vgg
from slim_module.preprocessing import vgg_preprocessing
import tensorflow.contrib.slim as slim

TRAINING_EXAMPLES_NUM = 20000
VALIDATION_EXAMPLES_NUM = 5000


def image_to_tfrecord(image_list, label_list, record_dir):
    writer = tf.python_io.TFRecordWriter(record_dir)
    for image, label in zip(image_list, label_list):
        with open(image, 'rb') as f:
            encoded_jpg = f.read()
        # with tf.gfile.GFile(image, 'rb') as fid:
        #     encoded_jpg = fid.read()
        # img = cv2.imread(image)
        # height, width, channel = img.shape
        # img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


def get_tfrecord_data(data_dir):
    image_list = []
    label_list = []

    for file in os.listdir(data_dir):
        name = file.split('.')
        image_list.append(os.path.join(data_dir, file))
        if name[0] == 'cat':
            label_list.append(0)
        else:
            label_list.append(1)

    tmp = np.array([image_list, label_list])
    tmp = tmp.transpose()
    # This function only shuffles the array along the first axis, so we need to transpose
    np.random.shuffle(tmp)
    image_list = list(tmp[:, 0])
    label_list = list(tmp[:, 1])
    label_list = [int(i) for i in label_list]
    with open('image_label_list.txt', 'w') as f:
        for i in range(len(image_list)):
            f.write(image_list[i] + '\t\t' + str(label_list[i]) + '\n')
    train_images = int(0.8 * len(image_list))
    image_to_tfrecord(image_list[:train_images], label_list[:train_images], './data/train_img.tfrecord')
    image_to_tfrecord(image_list[train_images:], label_list[train_images:], './data/validation_img.tfrecord')
    return image_list, label_list


def read_record(record_dir):
    for serialized_exam in tf.python_io.tf_record_iterator(record_dir):
        features = {'img_raw': tf.FixedLenFeature([], tf.string, ''),
                    'label': tf.FixedLenFeature([], tf.int64, 0)}
        parsed_features = tf.parse_single_example(serialized_exam, features)
        image = tf.image.decode_jpeg(parsed_features['img_raw'], channels=3)
        label = tf.cast(parsed_features['label'], tf.int64)
        with tf.Session() as sess:
            image, label = sess.run([image, label])
        # example = tf.train.Example()
        # example.ParseFromString(serialized_exam)
        #
        # image = example.features.feature['img_raw'].bytes_list.value[0]
        # label = example.features.feature['label'].int64_list.value[0]
        # image = np.asarray(bytearray(image), dtype=np.uint8)
        #
        # image = cv2.imdecode(image, flags=cv2.IMREAD_COLOR)
        cv2.imshow('image', image)
        cv2.waitKey(1000)

        print(image.shape, label)
    cv2.destroyAllWindows()


def preprocess(image, image_height, image_width, is_training):
    processed_image = vgg_preprocessing.preprocess_image(image, image_height, image_width, is_training)
    return processed_image


def parse_and_preprocess_data(example_proto, image_height, image_width, is_training):
    features = {'img_raw': tf.FixedLenFeature([], tf.string, ''),
                'label': tf.FixedLenFeature([], tf.int64, 0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.image.decode_jpeg(parsed_features['img_raw'], channels=3)
    label = tf.cast(parsed_features['label'], tf.int64)
    image = tf.cast(image, tf.float32)
    processed_image = preprocess(image, image_height, image_width, is_training)
    return processed_image, label


def inference(processed_images, class_num, is_training):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        net, endpoints = vgg.vgg_16(processed_images, num_classes=None, is_training=is_training)
    net = tf.squeeze(net, [1, 2])
    logits = slim.fully_connected(net, num_outputs=class_num, activation_fn=None)
    return logits


def loss(logits, labels):
    tf.losses.sparse_softmax_cross_entropy(labels, logits)
    loss = tf.losses.get_total_loss()
    return loss


def model_fn(features, labels, mode, params):
    logits = inference(features, params['class_num'], mode == tf.estimator.ModeKeys.TRAIN)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        # exclude = [v.name for v in tf.global_variables() if 'Adam' in v.name]
        exclude = ['fully_connected/weights', 'fully_connected/biases']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        tf.train.init_from_checkpoint('./model/vgg_16.ckpt', {v.name.split(':')[0]: v for v in variables_to_restore})

        global_step = tf.train.get_global_step()
        opt = tf.train.AdamOptimizer(0.0001)
        grad = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(grad, global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"eval_accuracy": accuracy}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def input_fn(filenames, batch_size, image_height, image_width, is_training):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda example:
                          parse_and_preprocess_data(example, image_height, image_width, is_training))

    dataset = dataset.batch(batch_size)
    if is_training:
        dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels
