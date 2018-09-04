import tensorflow as tf
import model_input
import cv2
import numpy as np
import os


def pred(test_data, log_dir):
    images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
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
            image = tf.cast(img, tf.float32)
            image = model_input.preprocess(image, 224, 224, False)
            imgs = tf.expand_dims(image, axis=0)
            imgs = imgs.eval()
            pre = sess.run(predict, feed_dict={images: imgs})

            if np.argmax(pre[0]) == 0:
                label = 'cat'
            else:
                label = 'dog'
            print('model:{}, file:{}, label: {}-{} ({:.2f}%)'.
                  format(ckpt.model_checkpoint_path, file, np.argmax(pre[0]), label, np.max(pre[0]) * 100))
            text = '{} {}({:.2f}%)'.format(f, label, np.max(pre[0]) * 100)
            cv2.putText(img, text, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('image', img)
            cv2.waitKey()
            count += 1


if __name__ == '__main__':
    pred('./data/test1', './log')