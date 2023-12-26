from keras import backend as K
from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
import cv2
import os


def scheduler(epoch, lr):
    return lr * 0.96 if epoch % 3 == 0 else lr


def wrapper(h: np.ndarray, ignore_class):
    def custom(y_true, y_pred):

        def infogain_loss(y_true, y_pred):
            true = K.eval(y_true)
            pred = K.eval(y_pred)
            batch_size = true.shape[0]
            loss = 0.
            for n in range(batch_size):
                for k in range(11):
                    if true[n][0] != ignore_class:
                        loss -= h[true[n][0]][k] * np.log(pred[n][k])

            loss /= batch_size
            print("\nloss:", loss)
            return loss

        loss = tf.py_function(infogain_loss, [y_true, y_pred], tf.float32)
        print("loss:", loss)
        loss.set_shape(())
        print("loss:", loss)
        return loss

    return custom


def augmentation(row, image, image_size, images_dataset, gt_dataset):
    shape = image.shape
    tolerance = ((shape[0] // 2 - image_size[0]) // 2, (shape[1] // 2 - image_size[1]) // 2)
    rw = row[1:].astype(float)
    row[-1] = "255.0"  # ignore label
    rw[2] = abs(rw[2])
    for i in range(int((np.sum(rw) * 2) ** 2)):
        delta = (
            random.randint(-tolerance[0], tolerance[0]) if tolerance[0] > 0 else 0,
            random.randint(-tolerance[1], tolerance[1]) if tolerance[1] > 0 else 0
        )
        center = shape[0] // 2, shape[1] // 2
        half = center[0] // 2, center[1] // 2

        crop_image = image[center[0] + delta[0] - half[0]:center[0] + delta[0] + half[0],
                           center[1] + delta[1] - half[1]:center[1] + delta[1] + half[1]]
        crop_image = cv2.resize(crop_image, image_size)

        flip_code = random.randint(-1, 2)
        flip_image = cv2.flip(crop_image, flip_code) if flip_code != 2 else crop_image

        images_dataset.append(np.array(cv2.resize(flip_image, image_size)))
        gt_dataset.append(row[1:].astype(np.float64))


def load_dataset(top_dir: str = "D:/SPBU/NIR/labour/dataset/test", image_size: tuple = (224, 224), aug=True) \
        -> tuple[np.ndarray, np.ndarray]:

    images_dataset, gt_dataset = [], []

    data = np.genfromtxt(f'{top_dir}/clear_ds.csv', delimiter=',', dtype=str)[1:]
    np.random.shuffle(data)
    print(data.shape)
    for row in data:
        try:
            image_path = row[0]
            image = cv2.imread(os.path.join(top_dir, image_path))

            images_dataset.append(np.array(cv2.resize(image, image_size)))
            gt_dataset.append(row[1:].astype(np.float64))

            if aug:
                augmentation(row, image, image_size, images_dataset, gt_dataset)

        except Exception as e:
            print(e)
            print(row)

    return np.array(images_dataset), np.array(gt_dataset)


def get_label_y(y: np.ndarray, output_layers: list[str]) \
        -> dict[str, np.ndarray]:

    array = np.full((len(y), len(y[0])), 0, dtype=int)

    for i in range(len(y)):
        for j in range(len(y[0])):
            if y[i][j] == 255.:
                array[i][j] = 255
                continue
            if i == 2:
                array[i][j] = int(((y[i][j] + 1.) / 2 + 0.05) * 10)
            else:
                array[i][j] = int((y[i][j] + 0.05) * 10)

    np.savetxt('ds.txt', array, fmt='%d')

    return dict(zip(output_layers, array))


def main():
    epoch = 200
    batch_size = 64
    img_height, img_width = 224, 224
    data_dir = "/home/mike/train"
    # data_dir = "D:/SPBU/NIR/labour/dataset/train"
    image_size = (img_height, img_width)

    # images, gt = load_dataset(top_dir=data_dir, image_size=image_size)
    # gt = gt.transpose()[:-1]
    # print(np.mean(gt, axis=1))
    # print(np.var(gt, axis=1))
    # print(np.std(gt, axis=1))
    #
    # images, gt = load_dataset(top_dir=data_dir, image_size=image_size, aug=False)
    # gt = gt.transpose()[:-1]
    # print(np.mean(gt, axis=1))
    # print(np.var(gt, axis=1))
    # print(np.std(gt, axis=1))
    images, gt = load_dataset(top_dir=data_dir, image_size=image_size, aug=True)
    defects = ["badExposure", "badWhiteBalance", "badSaturation", "noise", "haze", "undesiredBlur", "badComposition"]
    output_layers = list(map(lambda x: f'output_{x}', defects))
    gt = get_label_y(gt.transpose(), output_layers, )
    for output_layer in output_layers:
        np.savetxt(f'gt_aug_{output_layer}.txt', gt[output_layer], fmt='%d')
    # exit()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./log/SGDe07b32aii')
    lr_callback = keras.callbacks.LearningRateScheduler(scheduler)
    model = tf.keras.models.load_model('model_classification_batchnorm.h5')
    model.summary()


    x_train = images[len(images)//5:]
    y_train = gt[len(images)//5:].transpose()
    x_val = images[:len(images)//5]
    y_val = gt[:len(images)//5].transpose()

    # def get_h(name):
    #     arr = np.load(f'./infogain_weight/{name}.npy')
    #     try:
    #         if name == "badSaturation":
    #             print("w")
    #         else:
    #             arr = arr.reshape((11, 11))
    #     except Exception as e:
    #         print(e)
    #         print(arr.shape)
    #
    #     return arr
    #
    # loss = list(
    #     map(
    #         lambda x: wrapper(h=x, ignore_class=255),
    #         [
    #             np.squeeze(np.load(f'./infogain_weight/{name}.npy')) for name in defects
    #         ]
    #     )
    # )
    # losses = dict(zip(output_layers, loss))

    losses = dict(
        zip(
            output_layers,
            [keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=255)] * len(output_layers)
        )
    )
    loss_weights = dict(zip(output_layers, [1.0] * len(output_layers)))
    metrics = dict(zip(output_layers, [["accuracy"]] * len(output_layers)))
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
                  loss=losses,
                  loss_weights=loss_weights,
                  metrics=metrics)

    model.fit(
        x=x_train,
        y=get_label_y(y_train, output_layers,),
        batch_size=batch_size,
        epochs=epoch,
        verbose=1,
        shuffle=True,
        callbacks=[tensorboard_callback, lr_callback],
        validation_data=(x_val, get_label_y(y_val, output_layers))
    )

    model.save("fitted_model_classification_batchnorm_train", save_format="h5")

    # print('\ngt data:', np.array([y_train[:, 0], y_train[:, 10], y_train[:, 20]]))
    # predict = np.squeeze(np.array(model.predict([x_train[0], x_train[10], x_train[20]])))
    # for y in [y_train, y_val]:
    #     for i in [10, 20, 30]:
    #         print('\ngt data:', y[:, i])
    #         predict = np.squeeze(np.array(model.predict(x[i])))
    #         formatted = np.array2string(predict, formatter={'float_kind': lambda x: f"{x:.4f}"})
    #         print('\nprediction:', formatted)
    #     print('\n')


if __name__ == '__main__':
    main()
