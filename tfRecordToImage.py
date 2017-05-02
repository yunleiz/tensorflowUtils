import tensorflow as tf
import skimage.io as io

from imageCoder import ImageCoder


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
