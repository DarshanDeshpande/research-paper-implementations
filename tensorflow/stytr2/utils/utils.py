import tensorflow as tf


def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def load_vgg():
    vgg19 = tf.keras.applications.VGG19(include_top=False, input_shape=(256, 256, 3))
    vgg19 = tf.keras.models.Model(vgg19.input, [l.output for l in vgg19.layers[1:]])
    return vgg19
