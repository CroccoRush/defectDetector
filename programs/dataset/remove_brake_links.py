import os
import pandas as pd



def remover(dest_path, source_path):
    dataset = pd.read_csv(source_path, sep=",", )
    imgs = set(os.listdir(dest_path))

    dataset = dataset.apply(
        lambda col: col.apply(
            lambda x: x.split("/")[-1],
        ) if col.name == "url" else col,
    )

    def get_crack_rows():
        crack = list()
        for indx, row in dataset.iterrows():
            if row.url not in imgs:
                crack.append(indx)
        return crack

    clear_ds = dataset.drop(
        get_crack_rows()
    )

    clear_ds.to_csv(dest_path + "\\clear_ds.csv", index=False)


def main():
    main_path = "F:\\SPBU\\НИР\\Работа\\dataset"
    test_path = main_path + "\\test"
    train_path = main_path + "\\train"
    test_dataset = test_path + "\\defect_testing_gt.csv"
    train_dataset = train_path + "\\defect_training_gt.csv"

    remover(test_path, test_dataset)
    remover(train_path, train_dataset)


if __name__ == "__main__":
    main()

