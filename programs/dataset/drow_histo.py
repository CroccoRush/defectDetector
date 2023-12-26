import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from PIL import Image
from tqdm import tqdm


def drawer(dest_path: str, source_path: str, bartype: str = "amount") -> None:
    dataset = pd.read_csv(source_path, sep=",")
    dataset.apply(
        lambda col: draw1d(col.name, col.values, dest_path, bartype) if col.name != "url" else draw2d(col.name, col.values, dest_path)
    )


def draw1d(name: str, array: list, destination: str, bartype: str = "amount") -> None:
    if min(array) < 0:
        fig, ax = plt.subplots(figsize=(20, 7))
        space_array = [-1.05, -0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
        plt.xticks(np.arange(-1, 1.01, step=0.1))
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        space_array = [-0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
        plt.xticks(np.arange(0, 1.01, step=0.1))

    if bartype == "percent":
        heights, bins = np.histogram(array, bins=space_array)
        percent = [i / sum(heights) * 100 for i in heights]
        plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
        ax.bar(bins[:-1], percent, width=0.1, align='edge')
    else:
        ax.hist(array, space_array)

    ax.set_xlabel('Severity of the defect')
    ax.set_ylabel('Number of images')
    ax.set_title(f'Histogram of distribution images by {name}.')

    fig.tight_layout()
    path = f'{destination}\\fig\\disc\\{bartype}\\{name.replace(" ", "_").lower()}.png'
    plt.savefig(path,)


def draw2d(name: str, array: list, destination: str) -> None:
    width, height = list(), list()
    return
    for url in tqdm(array):
        # print(url)
        im = Image.open(f'{destination}\\{url}')
        w, h = im.size
        # print(w, h)
        width.append(w)
        height.append(h)
    plt.style.use('_mpl-gallery-nogrid')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(width, height)
    # ax.hist2d(width, height, bins=(np.arange(0, max(width)+1, 100), np.arange(0, max(height)+1, 100)))
    plt.xticks(np.arange(0, max(width)+100, step=100))
    plt.yticks(np.arange(0, max(height)+100, step=100))
    ax.set(xlim=(0, max(width)+100), ylim=(0, max(height)+100))

    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title(f'Histogram of images shape.')

    fig.tight_layout()
    # plt.show()
    plt.savefig(f'{destination}\\fig\\disc\\shape.png',)


def main():
    # main_path = "D:\\SPBU\\НИР\\labour\\dataset"
    # test_path = main_path + "\\test"
    # train_path = main_path + "\\train"
    # test_dataset = test_path + "\\clear_ds.csv"
    # train_dataset = train_path + "\\clear_ds.csv"

    main_path = "D:\\SPBU\\NIR\\labour\\programs\\pythonProject"
    train_path = main_path
    # defects = ["badExposure", "badWhiteBalance", "badSaturation", "noise", "haze", "undesiredBlur"]
    defects = ["badSaturation"]
    for defect in defects:
        train_dataset = train_path + f"\\gt_aug_output_{defect}.txt"

        array = np.genfromtxt(train_dataset, dtype=int).transpose()
        print(array)
        # drawer(train_path, train_dataset, bartype)"
        # drawer(train_path, train_dataset, bartype)

        fig, ax = plt.subplots(figsize=(10, 7))
        space_array = [-0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
        plt.xticks(np.arange(-10, 11, step=1))
        # plt.yticks(np.arange(0, 6000, step=1000))
        # plt.ylim(0, 6000)

        n, bin, patches = plt.hist(array, bins=21)
        # ax.hist(array, np.linspace(0, 12, 1))
        # plt.hist(array, bins=int(180 / 5))

        ax.set_xlabel('Severity of the defect')
        ax.set_ylabel('Number of images')
        ax.set_title(f'Histogram of distribution images by {defect} after augmentation.')

        fig.tight_layout()
        path = f'{train_path}\\{defect}_aug.png'
        plt.savefig(path, )


if __name__ == "__main__":
    main()

