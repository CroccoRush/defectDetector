import pandas as pd
import urllib.request


def parser(dest_path, source_path):
    dataset = pd.read_csv(source_path, sep=",", usecols=("url", )).values.transpose()[0]
    print(dataset)
    with open(f'{dest_path}\\{"crack_urls.txt"}', 'w+') as f:
        for url in dataset:
            name = str(url).split('/')[-1]
            try:
                a, b = urllib.request.urlretrieve(url, f'{dest_path}\\{name}')
            except Exception as e:
                print(f'{url}: {e}')
                f.write(url + '\n')

def main():
    main_path = "F:\\SPBU\\НИР\\Работа\\dataset"
    test_path = main_path + "\\test"
    train_path = main_path + "\\train"
    test_dataset = test_path + "\\defect_testing_gt.csv"
    train_dataset = train_path + "\\defect_training_gt.csv"

    parser(test_path, test_dataset)
    parser(train_path, train_dataset)


if __name__ == "__main__":
    main()

