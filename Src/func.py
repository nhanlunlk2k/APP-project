import os
import urllib.request
import tarfile

import pandas as pd
import pickle
import numpy as np
from numpy.lib.stride_tricks import as_strided

import matplotlib.pyplot as plt
import random
import math
import time

# Script này tải xuống và giải nén tập dữ liệu CIFAR-10 từ trang web chính thức
def download_cifar10():
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'Data/cifar-10-python.tar.gz'
    folder = 'Data/cifar-10-batches-py'

    if not os.path.exists(filename):
        print("⏬ Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, filename)

    if not os.path.exists(folder):
        print("📦 Extracting CIFAR-10...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
    print("✅ Done.")

# Load CIFAR-10 dataset
def load_batch(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        images = data[b'data']
        labels = data[b'labels']
        images = images.reshape(-1, 3, 32, 32)  # N x C x H x W
        return images, labels

def load_cifar10_data():
    base_dir = 'Data/cifar-10-batches-py'
    X_train, y_train = [], []

    # Load 5 training batches
    for i in range(1, 6):
        images, labels = load_batch(f'{base_dir}/data_batch_{i}')
        X_train.append(images)
        y_train += labels

    # Load test batch
    X_test, y_test = load_batch(f'{base_dir}/test_batch')

    # Convert to numpy arrays
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test

# in các ảnh ngẫu nhiên
def show_random_images(X, y, num_images=5):
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        idx = random.randint(0, len(X) - 1)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(X[idx].transpose(1, 2, 0))  # Chuyển đổi từ (C, H, W) sang (H, W, C)
        plt.title(f"Label: {y[idx]}")
        plt.axis('off')
    plt.show()

# in cây thư mục hiện tại
def print_directory_tree(path='.'):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

def log_training_details(epoch, epochs, epoch_time, n_samples, loss, accuracy, filename):
    # Tạo file nếu chưa có
    if not os.path.exists(filename):
        with open(filename, "w") as file:
            file.write("Epoch,Total Epochs,Epoch Time (s),Time per Step (ms),Loss,Accuracy\n")

    # Đọc toàn bộ nội dung
    with open(filename, "r") as file:
        lines = file.readlines()

    # Tính thời gian mỗi bước
    time_per_step = (epoch_time * 1000) / n_samples

    new_line = f"{epoch+1},{epochs},{epoch_time:.6f},{time_per_step:.6f},{loss:.6f},{accuracy:.6f}\n"

    # Kiểm tra và cập nhật dòng tương ứng với epoch
    updated = False
    for i in range(1, len(lines)):
        if lines[i].startswith(f"{epoch+1},"):
            lines[i] = new_line
            updated = True
            break

    if not updated:
        lines.append(new_line)

    # Ghi lại toàn bộ file
    with open(filename, "w") as file:
        file.writelines(lines)

def log_details(id, n_epochs,
                conv2d_time, relu_time, maxpool_time,
                flatten_time, fc_time, softmax_time,
                filename):

    if not os.path.exists(filename):
        with open(filename, "w") as file:
            file.write("ID,Total Epochs,Conv2D Time (s),ReLU Time (s),MaxPool2D Time (s),Flatten Time (s),FullyConnected Time (s),Softmax Time (s)\n")

    with open(filename, "r") as file:
        lines = file.readlines()

    new_line = f"{id},{n_epochs}," \
               f"{conv2d_time:.6f},{relu_time:.6f},{maxpool_time:.6f}," \
               f"{flatten_time:.6f},{fc_time:.6f},{softmax_time:.6f}\n"

    updated = False
    for i in range(1, len(lines)):
        if lines[i].startswith(f"{id},"):
            lines[i] = new_line
            updated = True
            break

    if not updated:
        lines.append(new_line)

    with open(filename, "w") as file:
        file.writelines(lines)

# vẽ biểu đồ cột plot thời gian chạy của các lớp được lưu trong details_log_python.csv
def plot_details(filename):
    with open(filename, "r") as file:
        lines = file.readlines()[1:]  # Bỏ qua tiêu đề
        data = [line.strip().split(",") for line in lines]

    ids = [d[0] for d in data]
    conv2d_times = [float(d[2]) for d in data]
    relu_times = [float(d[3]) for d in data]
    maxpool_times = [float(d[4]) for d in data]
    flatten_times = [float(d[5]) for d in data]
    fc_times = [float(d[6]) for d in data]
    softmax_times = [float(d[7]) for d in data]

    x = np.arange(len(ids))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 2 * width, conv2d_times, width, label='Conv2D')
    ax.bar(x - width, relu_times, width, label='ReLU')
    ax.bar(x, maxpool_times, width, label='MaxPool2D')
    ax.bar(x + width, flatten_times, width, label='Flatten')
    ax.bar(x + 2 * width, fc_times, width, label='FullyConnected')
    ax.bar(x + 3 * width, softmax_times, width, label='Softmax')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Time (s)')
    ax.set_title('Layer Execution Times')
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.legend()

    # ghi thời gian chạy của các lớp để dễ thể hiện lên biểu đồ vì có các lớp có thời gian chạy rất nhỏ
    for i in range(len(ids)):
        ax.text(x[i] - 2 * width, conv2d_times[i], f"{conv2d_times[i]:.2f}", ha='center', va='bottom')
        ax.text(x[i] - width, relu_times[i], f"{relu_times[i]:.2f}", ha='center', va='bottom')
        ax.text(x[i], maxpool_times[i], f"{maxpool_times[i]:.2f}", ha='center', va='bottom')
        ax.text(x[i] + width, flatten_times[i], f"{flatten_times[i]:.2f}", ha='center', va='bottom')
        ax.text(x[i] + 2 * width, fc_times[i], f"{fc_times[i]:.2f}", ha='center', va='bottom')
        ax.text(x[i] + 3 * width, softmax_times[i], f"{softmax_times[i]:.2f}", ha='center', va='bottom')

    plt.show()

# hàm plot theo epoch cho từng cột
def plot_timeline_from_log(csv_path, columns=None, title="Timeline", ylabel="Time (s)"):

    df = pd.read_csv(csv_path)

    # Nếu không truyền gì thì mặc định chọn tất cả các cột thời gian
    if columns is None:
        columns = [col for col in df.columns if "Time" in col and col != "Total Time (s)"]

    epochs = list(range(1, len(df) + 1))

    plt.figure(figsize=(10, 6))
    for col in columns:
        plt.plot(epochs, df[col], marker='o', label=col)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_compare_details_customized(file1, file2, label1='NumPy', label2='Numba'):
    def read_data(filename):
        with open(filename, "r") as file:
            lines = file.readlines()[1:]
            data = [line.strip().split(",") for line in lines]

        conv2d_times = [float(d[2]) for d in data]
        relu_times = [float(d[3]) for d in data]
        maxpool_times = [float(d[4]) for d in data]
        flatten_times = [float(d[5]) for d in data]
        fc_times = [float(d[6]) for d in data]
        softmax_times = [float(d[7]) for d in data]
        return [conv2d_times[0], relu_times[0], maxpool_times[0], flatten_times[0], fc_times[0], softmax_times[0]]

    # Lấy thời gian từng lớp từ hai file
    times1 = read_data(file1)
    times2 = read_data(file2)
    layer_names = ['Conv2D', 'ReLU', 'MaxPool2D', 'Flatten', 'FC', 'Softmax']

    x = np.arange(len(layer_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, times1, width, label=label1, color='skyblue')
    bars2 = ax.bar(x + width/2, times2, width, label=label2, color='orange')

    # Ghi giá trị lên từng cột
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Time (s)')
    ax.set_title('Execution Time Comparison per Layer in 1st Epoch')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def preprocess_cifar10_data():
    X_train, y_train, X_test, y_test = load_cifar10_data()

    # Chuyển từ (N, C, H, W) -> (N, H, W, C) và chuẩn hóa về [0, 1]
    X_train = X_train.transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    X_test = X_test.transpose(0, 2, 3, 1).astype(np.float32) / 255.0

    # One-hot encoding cho nhãn
    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]

    return X_train, y_train_onehot, X_test, y_test_onehot