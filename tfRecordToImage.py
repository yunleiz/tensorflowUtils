import tensorflow as tf
import numpy as np
import skimage.io as io


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def image_label(filepath, coder):
    record_iterator = tf.python_io.tf_record_iterator(path=filepath)
    image = None
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['image/height']
                     .int64_list
                     .value[0])

        width = int(example.features.feature['image/width']
                    .int64_list
                    .value[0])

        depth = int(example.features.feature['image/channels']
                    .int64_list
                    .value[0])

        label = (example.features.feature['image/class/label']
                 .bytes_list
                 .value[0])

        img_string = (example.features.feature['image/encoded']
                      .bytes_list
                      .value[0])

        image = coder.decode_jpeg(img_string)
        yield image, label


def main(unused_argv):
    tfrecords_filename = './tfrecord/test.tfrecords'
    counter = 0
    for image, label in image_label(tfrecords_filename, ImageCoder()):
        io.imshow(image)
        io.show()
        print(label)
        counter += 1
        if counter >= 3:
            break

if __name__ == '__main__':
    tf.app.run()
