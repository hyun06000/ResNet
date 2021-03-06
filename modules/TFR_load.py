import os
import tensorflow as tf

def random_flip_and_crop(example):
    print(example)
    print(type(example))
    
    
    image, label = example["image"], example["label"]
    image = tf.image.random_flip_left_right(image)
    image = tf.image.pad_to_bounding_box(
        image,
        offset_height = 2,
        offset_width  = 2,
        target_height = 32 + 4,
        target_width  = 32 + 4
    )
    image = tf.image.random_crop(
        value = image,
        size  = (32, 32, 3),
    )

    return {"image" : image, "label" : label}

class decode_fn:
        def __init__(self, is_training, data_shape):
            self.is_training = is_training
            self.data_shape = data_shape

        def __call__(self, record_bytes):
            example = tf.io.parse_single_example(
                # Data
                record_bytes,

                # Schema
                {"image": tf.io.FixedLenFeature([], dtype=tf.string),
                 "label": tf.io.FixedLenFeature([], dtype=tf.int64)}
            )
            image = tf.io.parse_tensor(example["image"],
                                       out_type = tf.uint8)
            image = tf.ensure_shape(image, self.data_shape)

            if self.is_training:
                # augmentation
                image = tf.image.random_flip_left_right(image)

            image = tf.cast(image, tf.float32) * 1./255

            label = example["label"]
            label = tf.one_hot(label,7)

            return image, label


def TFR_load(
    path,
    BATCH_SIZE,
    NUM_TRAIN_DATA,
    is_training = False,
    data_shape = [227, 227, 3]
):
    
    files = [path+n for n in os.listdir(path)]
    ds = tf.data.TFRecordDataset(files)\
                    .map(decode_fn(is_training, data_shape))\
                    .repeat()\
                    .shuffle(NUM_TRAIN_DATA, reshuffle_each_iteration=True)\
                    .batch(BATCH_SIZE)\
                    .prefetch(tf.data.AUTOTUNE)
    return ds