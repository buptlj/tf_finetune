import tensorflow as tf
import model_input
import cv2
import numpy as np
import os


def pred(test_data, log_dir):
    # img = cv2.imread('./data/test1/' + '1.jpg')
    images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    # img = tf.cast(img, tf.float32)
    # img = model_input.preprocess(img, 224, 224, False)
    # images = tf.expand_dims(img, axis=0)
    logits = model_input.inference(images, 2, False)
    predict = tf.nn.softmax(logits)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('no checkpoint file')
            return
        count = 0
        for f in os.listdir(test_data):
            if count >= 10:
                break
            file = os.path.join(test_data, f)
            img = cv2.imread(file)
            img = tf.cast(img, tf.float32)
            img = model_input.preprocess(img, 224, 224, False)
            imgs = tf.expand_dims(img, axis=0)
            imgs = imgs.eval()
            pre = sess.run(predict, feed_dict={images: imgs})
            for p in pre:
                if np.argmax(p) == 0:
                    label = 'cat'
                else:
                    label = 'dog'
                print('model:{}, file:{}, label: {}-{} ({:.2f}%)'.
                      format(ckpt.model_checkpoint_path, file, np.argmax(p), label, np.max(p) * 100))
            count += 1


if __name__ == '__main__':
    # imgs = np.zeros(shape=(10, 224, 224, 3))
    # i = 0
    #
    # for f in os.listdir('./data/test1'):
    #
    #     if i >= 10:
    #         break
    #     img = cv2.imread('./data/test1/'+f)
    #     img = cv2.resize(img, (224, 224))
    #     imgs[i] = img
    #     i += 1
    #
    #     # imgs = tf.expand_dims(img, axis=0)
    pred('./data/test1', './log')