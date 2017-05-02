# tensorflowUtils

## image_tfrecord converter

### image to tfrecord
This code is just for convert all images from folders to tfrecord
with the parent folder name as the label. Those tfrecords will be
used for train the google inception model, so the feature map has
to match the feature map required in the inception training, which
is:
```python
'image/height': tf.int64
'image/width': tf.int64
'image/colorspace': 'RGB'
'image/channels': default 3
'image/class/label': tf.int64 but stupid let's try tf.string first'
'image/class/synset': no clue set to null
'image/class/text': tf.string the description of the image
'image/object/bbox/xmin': no clue set to null
'image/object/bbox/xmax': no clue set to null
'image/object/bbox/ymin': no clue set to null
'image/object/bbox/ymax': no clue set to null
'image/object/bbox/label': no clue set to null
'image/format': defualt 'JPEG'
'image/filename': try null first, not important
'image/encoded': <JPEG encoded string>
```


### tfrecord to image
Just a validation tool for the image to tfrecord. Should not be used in the train process