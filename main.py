from tensorflow import keras
from keras import layers
import tensorflow as tf
import numpy as np


def inception_3a(previous_layer):
    branch_0 = layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_3a_Conv_1x1')(previous_layer)
    branch_0 = layers.BatchNormalization()(branch_0)

    branch_1 = layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_3a_Conv_3x3_reduce')(previous_layer)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_3a_Conv_3x3')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1 )

    branch_2 = layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_3a_Conv_double_3x3_reduce')(previous_layer)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=96, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_3a_Conv_double_3x3a')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=96, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_3a_Conv_double_3x3b')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)

    branch_3 = layers.AveragePooling2D(pool_size=[3, 3], strides=(1, 1), padding='same',
                                       name='Inception_3a_AvePool_3x3')(previous_layer)
    branch_3 = layers.Conv2D(filters=32, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_3a_Conv_1x1_pool_porj')(branch_3)
    branch_3 = layers.BatchNormalization()(branch_3)

    # Concat
    return tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3, name='Inception_3a_output')


def inception_3b(previous_layer):
    branch_0 = layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_3b_Conv_1x1')(previous_layer)
    branch_0 = layers.BatchNormalization()(branch_0)

    branch_1 = layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_3b_Conv_3x3_reduce')(previous_layer)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=96, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_3b_Conv_3x3')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1)

    branch_2 = layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_3b_Conv_double_3x3_reduce')(previous_layer)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=96, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_3b_Conv_double_3x3a')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=96, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_3b_Conv_double_3x3b')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)

    branch_3 = layers.AveragePooling2D(pool_size=[3, 3], strides=(1, 1), padding='same',
                                       name='Inception_3b_AvePool_3x3')(previous_layer)
    branch_3 = layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_3b_Conv_1x1_pool_porj')(branch_3)
    branch_3 = layers.BatchNormalization()(branch_3)

    # Concat
    return tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3, name='Inception_3b_output')


def inception_3c(previous_layer):
    branch_0 = layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_3c_Conv_3x3_reduce')(previous_layer)
    branch_0 = layers.BatchNormalization()(branch_0)
    branch_0 = layers.Conv2D(filters=160, kernel_size=[3, 3], strides=(2, 2), padding='same', activation='relu',
                             name='Inception_3c_Conv_3x3')(branch_0)
    branch_0 = layers.BatchNormalization()(branch_0)

    branch_1 = layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_3c_Conv_double_3x3_reduce')(previous_layer)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=96, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_3c_Conv_double_3x3a')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=96, kernel_size=[3, 3], strides=(2, 2), padding='same', activation='relu',
                             name='Inception_3c_Conv_double_3x3b')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1)

    branch_2 = layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2), padding='same',
                                   name='Inception_3c_MaxPool_3x3')(previous_layer)

    # Concat
    return tf.concat(values=[branch_0, branch_1, branch_2], axis=3, name='Inception_3c_output')


def inception_4a(previous_layer):
    branch_0 = layers.Conv2D(filters=224, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4a_Conv_1x1')(previous_layer)
    branch_0 = layers.BatchNormalization()(branch_0)

    branch_1 = layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4a_Conv_3x3_reduce')(previous_layer)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=96, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4a_Conv_3x3')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1)

    branch_2 = layers.Conv2D(filters=96, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4a_Conv_double_3x3_reduce')(previous_layer)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4a_Conv_double_3x3a')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4a_Conv_double_3x3b')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)

    branch_3 = layers.AveragePooling2D(pool_size=[3, 3], strides=(1, 1), padding='same',
                                       name='Inception_4a_AvePool_3x3')(previous_layer)
    branch_3 = layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4a_Conv_1x1_pool_porj')(branch_3)
    branch_3 = layers.BatchNormalization()(branch_3)

    # Concat
    return tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3, name='Inception_4a_output')


def inception_4b(previous_layer):
    branch_0 = layers.Conv2D(filters=192, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4b_Conv_1x1')(previous_layer)
    branch_0 = layers.BatchNormalization()(branch_0)

    branch_1 = layers.Conv2D(filters=96, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4b_Conv_3x3_reduce')(previous_layer)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4b_Conv_3x3')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1)

    branch_2 = layers.Conv2D(filters=96, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4b_Conv_double_3x3_reduce')(previous_layer)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4b_Conv_double_3x3a')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4b_Conv_double_3x3b')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)

    branch_3 = layers.AveragePooling2D(pool_size=[3, 3], strides=(1, 1), padding='same',
                                       name='Inception_4b_AvePool_3x3')(previous_layer)
    branch_3 = layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4b_Conv_1x1_pool_porj')(branch_3)
    branch_3 = layers.BatchNormalization()(branch_3)

    # Concat
    return tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3, name='Inception_4b_output')


def inception_4c(previous_layer):
    branch_0 = layers.Conv2D(filters=160, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4c_Conv_1x1')(previous_layer)
    branch_0 = layers.BatchNormalization()(branch_0)

    branch_1 = layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4c_Conv_3x3_reduce')(previous_layer)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=160, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4c_Conv_3x3')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1)

    branch_2 = layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4c_Conv_double_3x3_reduce')(previous_layer)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=160, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4c_Conv_double_3x3a')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=160, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4c_Conv_double_3x3b')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)

    branch_3 = layers.AveragePooling2D(pool_size=[3, 3], strides=(1, 1), padding='same',
                                       name='Inception_4c_AvePool_3x3')(previous_layer)
    branch_3 = layers.Conv2D(filters=96, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4c_Conv_1x1_pool_porj')(branch_3)
    branch_3 = layers.BatchNormalization()(branch_3)

    # Concat
    return tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3, name='Inception_4c_output')


def inception_4d(previous_layer):
    branch_0 = layers.Conv2D(filters=96, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4d_Conv_1x1')(previous_layer)
    branch_0 = layers.BatchNormalization()(branch_0)

    branch_1 = layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4d_Conv_3x3_reduce')(previous_layer)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=192, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4d_Conv_3x3')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1)

    branch_2 = layers.Conv2D(filters=160, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4d_Conv_double_3x3_reduce')(previous_layer)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=192, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4d_Conv_double_3x3a')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=192, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4d_Conv_double_3x3b')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)

    branch_3 = layers.AveragePooling2D(pool_size=[3, 3], strides=(1, 1), padding='same',
                                       name='Inception_4d_AvePool_3x3')(previous_layer)
    branch_3 = layers.Conv2D(filters=96, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4d_Conv_1x1_pool_porj')(branch_3)
    branch_3 = layers.BatchNormalization()(branch_3)

    # Concat
    return tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3, name='Inception_4d_output')


def inception_4e(previous_layer):
    branch_0 = layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4e_Conv_3x3_reduce')(previous_layer)
    branch_0  = layers.BatchNormalization()(branch_0)
    branch_0 = layers.Conv2D(filters=192, kernel_size=[3, 3], strides=(2, 2), padding='same', activation='relu',
                             name='Inception_4e_Conv_3x3')(branch_0)
    branch_0 = layers.BatchNormalization()(branch_0)

    branch_1 = layers.Conv2D(filters=192, kernel_size=[1, 1], padding='same', activation='relu',
                             name='Inception_4e_Conv_double_3x3_reduce')(previous_layer)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation='relu',
                             name='Inception_4e_Conv_double_3x3a')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=256, kernel_size=[3, 3], strides=(2, 2), padding='same', activation='relu',
                             name='Inception_4e_Conv_double_3x3b')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1)

    branch_2 = layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2), padding='same',
                                   name='Inception_4e_AvePool_3x3')(previous_layer)

    # Concat
    return tf.concat(values=[branch_0, branch_1, branch_2], axis=3, name='Inception_4e_output')


def inception_5a(previous_layer, defect_type):
    branch_0 = layers.Conv2D(filters=352, kernel_size=[1, 1], padding='same', activation='relu',
                             name=f'Inception_5a_Conv_1x1_{defect_type}')(previous_layer)
    branch_0 = layers.BatchNormalization()(branch_0)

    branch_1 = layers.Conv2D(filters=192, kernel_size=[1, 1], padding='same', activation='relu',
                             name=f'Inception_5a_Conv_3x3_reduce_{defect_type}')(previous_layer)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=320, kernel_size=[3, 3], padding='same', activation='relu',
                             name=f'Inception_5a_Conv_3x3_{defect_type}')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1)

    branch_2 = layers.Conv2D(filters=160, kernel_size=[1, 1], padding='same', activation='relu',
                             name=f'Inception_5a_Conv_double_3x3_reduce_{defect_type}')(previous_layer)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=224, kernel_size=[3, 3], padding='same', activation='relu',
                             name=f'Inception_5a_Conv_double_3x3a_{defect_type}')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Conv2D(filters=224, kernel_size=[3, 3], padding='same', activation='relu',
                             name=f'Inception_5a_Conv_double_3x3b_{defect_type}')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)

    branch_3 = layers.AveragePooling2D(pool_size=[3, 3], strides=(1, 1), padding='same',
                                       name=f'Inception_5a_AvePool_3x3_{defect_type}')(previous_layer)
    branch_3 = layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation='relu',
                             name=f'Inception_5a_Conv_1x1_pool_porj_{defect_type}')(branch_3)
    branch_2 = layers.BatchNormalization()(branch_2)

    # Concat
    return tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=3, name=f'Inception_5a_output_{defect_type}')


def inception_5b(previous_layer, defect_type):
    branch_0 = layers.Conv2D(filters=88, kernel_size=[1, 1], padding='same', activation='relu',
                             name=f'Inception_5b_Conv_1x1_{defect_type}')(previous_layer)
    branch_0 = layers.BatchNormalization()(branch_0)

    branch_1 = layers.Conv2D(filters=192, kernel_size=[1, 1], padding='same', activation='relu',
                             name=f'Inception_5b_Conv_double_3x3_reduce_{defect_type}')(previous_layer)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=224, kernel_size=[3, 3], padding='same', activation='relu',
                             name=f'Inception_5b_Conv_double_3x3a_{defect_type}')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Conv2D(filters=56, kernel_size=[3, 3], padding='same', activation='relu',
                             name=f'Inception_5b_Conv_double_3x3b_{defect_type}')(branch_1)
    branch_1 = layers.BatchNormalization()(branch_1)

    branch_2 = layers.MaxPooling2D(pool_size=[3, 3], strides=(1, 1), padding='same',
                                   name=f'Inception_5b_AvePool_3x3_{defect_type}')(previous_layer)
    branch_2 = layers.Conv2D(filters=32, kernel_size=[1, 1], padding='same', activation='relu',
                             name=f'Inception_5b_Conv_1x1_pool_porj_{defect_type}')(branch_2)
    branch_2 = layers.BatchNormalization()(branch_2)

    # Concat
    return tf.concat(values=[branch_0, branch_1, branch_2], axis=3, name=f'Inception_5b_output_{defect_type}')


def inception_changed(input_shape):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(filters=64, kernel_size=[7, 7], strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=[3, 3], strides=2, padding='same')(x)
    x = layers.Conv2D(filters=64, kernel_size=[1, 1], strides=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=192, kernel_size=[3, 3], strides=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=[3, 3], strides=2, padding='same')(x)

    x = inception_3a(x)
    x = inception_3b(x)
    x = inception_3c(x)

    x = inception_4a(x)
    x = inception_4b(x)
    x = inception_4c(x)
    x = inception_4d(x)
    x = inception_4e(x)

    defects = ["badComposition", "undesiredBlur", "haze", "noise", "badSaturation", "badWhiteBalance", "badExposure"]
    outputs = list()
    for defect in defects:
        output = inception_5a(x, defect)
        output = inception_5b(output, defect)

        output = keras.layers.AveragePooling2D(pool_size=(7, 7), strides=1, name=f"pool5_7x7_s1_{defect}")(output)
        output = keras.layers.Dropout(rate=0.5, name=f"pool5_drop_7x7_s1_{defect}")(output)
        output = keras.layers.Dense(units=128, activation="relu", name=f"loss3_new_classifier_{defect}")(output)
        output = keras.layers.Dropout(rate=0.5, name=f"drop_{defect}")(output)
        output = keras.layers.Dense(units=11, activation=None, name=f"loss3_new_new_classifier_{defect}")(output)
        output = keras.layers.Reshape(target_shape=(11,), name=f"reshape_{defect}")(output)
        output = keras.layers.Activation("softmax", name=f"output_{defect}")(output)
        outputs.append(output)

    return keras.Model(inputs, outputs, name="Inception_changed")


class VOV(keras.Model):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.infogain_loss_tracker = list()
        self.defects = ["badComposition", "undesiredBlur", "haze", "noise",
                        "badSaturation", "badWhiteBalance", "badExposure"]

        for defect in self.defects:
            self.infogain_loss_tracker.append(keras.metrics.Mean(name=f"infogain_loss_{defect}"))

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        shape = data[0].shape
        with tf.GradientTape() as tape:
            z = self.model(data)  # (7, batch_size, 1, 1, 11)
            # loss = -1/shape[1] * tf.reduce_sum(
            #     tf.reduce_sum(
            #         np.log(z, axis=-1),
            #         axis=-1
            #     ),
            #     axis=1
            # )

            losses = list()
            for d in range(len(self.defects)):
                loss = 0
                for n in range(shape[1]):
                    for k in range(11):
                        loss += h[d, n, k] * np.log(z[d, n, 1, 1, k])  # h - matrix of probability
                losses.append(-loss / shape[0])

        grads = tape.gradient(losses, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        for d in range(len(self.defects)):
            self.infogain_loss_tracker[d].update_state(losses[d])

        loss_result = list()
        for tracker in self.infogain_loss_tracker:
            loss_result.append(tracker.result())

        return dict(zip(self.defects, loss_result))


def main():
    img_height, img_width = 224, 224
    image_size = (img_height, img_width)
    input_shape = image_size + (3,)

    model = inception_changed(input_shape)
    model.save("model_classification_batchnorm.h5")


if __name__ == '__main__':
    main()
