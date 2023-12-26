import os

import numpy as np
import cv2
import tensorflow as tf


def main():
    img_height, img_width = 224, 224
    image_size = (img_height, img_width)

    def load_dataset(top_dir: str = "D:/SPBU/NIR/labour/dataset/test") -> tuple[np.ndarray, np.ndarray]:
        images_dataset, gt_dataset = [], []

        data = np.genfromtxt(f'{top_dir}/clear_ds.csv', delimiter=',', dtype=str)
        np.random.shuffle(data)
        for row in data:
            try:
                image_path = row[0]
                image = cv2.imread(os.path.join(top_dir, image_path))
                image = cv2.resize(image, image_size)

                images_dataset.append(np.array(image))
                gt_dataset.append(row[1:].astype(np.float64))

            except Exception as e:
                print(e)
                print(row)

        return np.array(images_dataset), np.array(gt_dataset)

    images, gt = load_dataset()

    x_test = images
    y_test = gt.transpose()

    model = tf.keras.models.load_model('fitted_model_classification_train')

    predictions = np.squeeze(np.array(model.predict(x_test[:3])))

    formatted = np.array2string(predictions, formatter={'float_kind': lambda x: f"{x:.4f}"})
    print('\ngt data:', y_test[:, :3])
    print('\nprediction:', formatted)

    peak = np.arange(0.0, 1.1, 0.1)
    peak = np.reshape(peak, (1, -1))
    my_peak = np.array(peak)
    my_peak = np.tile(my_peak, (3, 1))
    score = np.sum(predictions * my_peak, 1)
    formatted_score = np.array2string(score, formatter={'float_kind': lambda x: f"{x:.4f}"})
    print('\nprediction_c:', formatted_score)


if __name__ == '__main__':
    main()
