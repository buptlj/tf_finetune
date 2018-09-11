import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.models.research.slim.nets import vgg
from tensorflow.models.research.slim.preprocessing import vgg_preprocessing
from tensorflow.models.research.slim.nets import inception_v3
from tensorflow.models.research.slim.preprocessing import inception_preprocessing
from tensorflow.models.research.slim.nets import resnet_v1
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


def preprocess(image, pre_trained_model, image_size, is_training):
    if ('vgg_16' in pre_trained_model) or ('resnet_v1_50' in pre_trained_model):
        processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training)
    elif 'inception_v3' in pre_trained_model:
        # processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training)
        image = tf.expand_dims(image, 0)
        processed_image = tf.image.resize_bilinear(image, [image_size, image_size])
        processed_image = tf.squeeze(processed_image)
        processed_image.set_shape([None, None, 3])
    else:
        print('wrong input pre_trained_model')
        return
    return processed_image


def parse_and_preprocess_data(example_proto, pre_trained_model, image_size, is_training):
    features = {'img_raw': tf.FixedLenFeature([], tf.string, ''),
                'label': tf.FixedLenFeature([], tf.int64, 0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.image.decode_jpeg(parsed_features['img_raw'], channels=3)
    label = tf.cast(parsed_features['label'], tf.int64)
    image = tf.cast(image, tf.float32)
    processed_image = preprocess(image, pre_trained_model, image_size, is_training)
    return processed_image, label


def inference(pre_trained_model, processed_images, class_num, is_training):
    if 'vgg_16' in pre_trained_model:
        print('load model: vgg_16')
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net, endpoints = vgg.vgg_16(processed_images, num_classes=None, is_training=is_training)
        net = tf.squeeze(net, [1, 2])
        logits = slim.fully_connected(net, num_outputs=class_num, activation_fn=None)
        # fc6 = endpoints['vgg_16/fc6']
        # net = tf.squeeze(fc6, [1, 2])
        # logits = slim.fully_connected(net, num_outputs=class_num, activation_fn=None)
    elif 'inception_v3' in pre_trained_model:
        print('load model: inception_v3')
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            net, endpoints = inception_v3.inception_v3_base(processed_images)
        kernel_size = inception_v3._reduced_kernel_size_for_small_input(net, [8, 8])
        net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                              scope='AvgPool_1a_{}x{}'.format(*kernel_size))
        net = tf.squeeze(net, [1, 2])
        logits = slim.fully_connected(net, num_outputs=class_num, activation_fn=None)
    elif 'resnet_v1_50' in pre_trained_model:
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits, endpoints = resnet_v1.resnet_v1_50(processed_images, class_num, is_training=is_training)
    else:
        print('wrong input pre_trained_model')
        return
    return logits


def loss(logits, labels):
    tf.losses.sparse_softmax_cross_entropy(labels, logits)
    loss = tf.losses.get_total_loss()
    return loss


def variables_to_restore_and_train(pre_trained_model):
    if 'vgg_16' in pre_trained_model:
        exclude = ['fully_connected']
        train_sc = ['fully_connected']
    elif 'inception_v3' in pre_trained_model:
        exclude = ['InceptionV3/Logits', 'InceptionV3/AuxLogits', 'fully_connected']
        train_sc = ['fully_connected']
    elif 'resnet_v1_50' in pre_trained_model:
        exclude = ['resnet_v1_50/logits']
        train_sc = ['resnet_v1_50/logits']
    else:
        exclude = []
        train_sc = []
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    variables_to_train = []
    for sc in train_sc:
        variables_to_train += slim.get_trainable_variables(sc)
    return variables_to_train, variables_to_restore


def get_train_op(total_loss, variables_to_train, variables_to_restore, batch_size, learning_rate, global_step):
    num_batches_per_epoch = TRAINING_EXAMPLES_NUM / batch_size
    decay_steps = int(num_batches_per_epoch)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(learning_rate=learning_rate,
                                    global_step=global_step,
                                    decay_steps=decay_steps,
                                    decay_rate=0.9,
                                    staircase=True)
    opt1 = tf.train.MomentumOptimizer(lr, momentum=0.9)
    opt2 = tf.train.MomentumOptimizer(0.01 * lr, momentum=0.9)
    # opt2 = tf.train.GradientDescentOptimizer(0.01 * lr)
    grads = tf.gradients(total_loss, variables_to_train + variables_to_restore)
    grads1 = grads[:len(variables_to_train)]
    grads2 = grads[len(variables_to_train):]
    train_op1 = opt1.apply_gradients(zip(grads1, variables_to_train), global_step)
    train_op2 = opt2.apply_gradients(zip(grads2, variables_to_restore))
    train_op = tf.group(train_op1, train_op2)

    return train_op


def model_fn(features, labels, mode, params):
    logits = inference(params['model_path'], features, params['class_num'], mode == tf.estimator.ModeKeys.TRAIN)
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
        variables_to_train, variables_to_restore = variables_to_restore_and_train(params['model_path'])
        tf.train.init_from_checkpoint(params['model_path'], {v.name.split(':')[0]: v for v in variables_to_restore})

        global_step = tf.train.get_or_create_global_step()
        train_op = get_train_op(loss, variables_to_train, variables_to_restore,
                                params['batch_size'], params['lr'], global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"eval_accuracy": accuracy}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def input_fn(filenames, batch_size, pre_trained_model, image_size, is_training):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda example:
                          parse_and_preprocess_data(example, pre_trained_model, image_size, is_training))

    dataset = dataset.batch(batch_size)
    if is_training:
        dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels
