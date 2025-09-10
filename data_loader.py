# sp/data_loader.py

import torch
import os
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, Subset

# 'wilds'ライブラリのインポート
try:
    import wilds
except ImportError:
    print("Warning: 'wilds' library not found. WaterBirds dataset will not be available.")
    print("Please install it using: pip install wilds")
    wilds = None

def colorize_mnist(images, labels, correlation):
    """ MNISTデータセットに色付けを行い，ラベルと色の相関を持つデータを作成する """
    labels_pm1 = (labels >= 5).float() * 2.0 - 1.0
    images_gray = images.float() / 255.0
    n_samples = len(labels_pm1)

    images_rgb = torch.stack([images_gray, images_gray, images_gray], dim=1)

    prob_color_matches_label = (1.0 + correlation) / 2.0
    attributes_pm1 = torch.zeros_like(labels_pm1)

    for i in range(n_samples):
        y_i = labels_pm1[i]
        if torch.rand(1) < prob_color_matches_label:
            a_i = y_i
        else:
            a_i = -y_i
        attributes_pm1[i] = a_i

    digit_mask = (images_gray > 0.01).unsqueeze(1)
    color_factors = torch.ones(n_samples, 3, 1, 1, dtype=images_gray.dtype)

    red_indices = (attributes_pm1 == 1.0)
    green_indices = (attributes_pm1 == -1.0)

    color_factors[red_indices, 1, :, :] = 0.25
    color_factors[red_indices, 2, :, :] = 0.25
    color_factors[green_indices, 0, :, :] = 0.25
    color_factors[green_indices, 2, :, :] = 0.25

    colored_images = images_rgb * color_factors
    final_images_rgb = torch.where(digit_mask, colored_images, images_rgb)

    return final_images_rgb, labels_pm1, attributes_pm1

def get_colored_mnist(num_samples, correlation, train=True):
    """ ColoredMNISTデータセットをロードして生成する """
    print(f"Preparing Colored MNIST for {'train' if train else 'test'} set...")
    mnist_dataset = MNIST('./data', train=train, download=True)

    images = mnist_dataset.data[:num_samples]
    targets = mnist_dataset.targets[:num_samples]

    return colorize_mnist(images, targets, correlation)

def get_waterbirds_dataset(num_train, num_test, image_size):
    """ WILDSライブラリからWaterBirdsデータセットをロードする """
    if wilds is None:
        raise ImportError("WaterBirds dataset requires the 'wilds' library. Please install it.")

    # 破損している可能性のあるアーカイブファイルを削除する
    dataset_archive_path = 'data/waterbirds_v1.0/archive.tar.gz'
    if os.path.exists(dataset_archive_path):
        print(f"Removing potentially corrupted archive: {dataset_archive_path}")
        try:
            os.remove(dataset_archive_path)
        except OSError as e:
            print(f"Error removing archive file: {e}")

    full_dataset = wilds.get_dataset(dataset='waterbirds', root_dir='data', download=True)
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    train_dataset = full_dataset.get_subset('train', transform=transform)
    train_indices = np.random.choice(len(train_dataset), num_train, replace=False) if num_train < len(train_dataset) else np.arange(len(train_dataset))
    train_subset = Subset(train_dataset, train_indices)

    test_dataset = full_dataset.get_subset('test', transform=transform)
    test_indices = np.random.choice(len(test_dataset), num_test, replace=False) if num_test < len(test_dataset) else np.arange(len(test_dataset))
    test_subset = Subset(test_dataset, test_indices)

    # DataLoaderを使って全データを一括で取得
    train_loader = DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=len(test_subset), shuffle=False)

    X_train, y_train_01, metadata_train = next(iter(train_loader))
    X_test, y_test_01, metadata_test = next(iter(test_loader))

    # ラベルを-1, +1形式に変換
    y_train_pm1 = y_train_01.float() * 2.0 - 1.0
    a_train_pm1 = metadata_train[:, 0].float() * 2.0 - 1.0 # place_of_birdが属性
    y_test_pm1 = y_test_01.float() * 2.0 - 1.0
    a_test_pm1 = metadata_test[:, 0].float() * 2.0 - 1.0

    return X_train, y_train_pm1, a_train_pm1, X_test, y_test_pm1, a_test_pm1
