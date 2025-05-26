# **Đồ án môn học**
## Môn: Lập trình song song ứng dụng
### Lớp CQ2021/21 - Nhóm 05
#### Thành viên:
| Họ và tên             | MSSV |
| :-----------          |     :----:|
| Diệp Đại Thiện Nhân | 18120491  |
| Hoàng Trung Nam  | 21120290 |

# **Tổng quan đồ án**

- Nhóm quyết định thử sức với VGG.

- VGG là một kiến trúc CNN sâu được giới thiệu bởi nhóm VGG (Visual Geometry Group) thuộc Đại học Oxford năm 2014. Kiến trúc này nổi bật bởi việc sử dụng nhiều lớp convolution 3x3 liên tiếp thay vì một lớp convolution lớn (vd: 5x5 hay 7x7), nhằm tăng khả năng học đặc trưng và giữ kích thước receptive field ổn định.

- Hai phiên bản phổ biến:
  - VGG-16: 16 lớp có trọng số (13 conv + 3 FC).
  - VGG-19: 19 lớp có trọng số (16 conv + 3 FC).

- **Input của CIFAR-10** là `32x32x3`, nhỏ hơn rất nhiều so với `224x224x3` của VGG gốc.

Vậy nên, khi áp dụng VGG cho CIFAR-10, ta cần **giảm số lượng layer** hoặc **điều chỉnh input size** như sau:
## **Kiến trúc mô hình cụ thể**

- **Input Layer**  
  - 32×32×3 (RGB)

- **Block 1**  
  - Conv3×3, 64 filters + ReLU  
  - Conv3×3, 64 filters + ReLU  
  - MaxPool 2×2

- **Block 2**  
  - Conv3×3, 128 filters + ReLU  
  - Conv3×3, 128 filters + ReLU  
  - MaxPool 2×2

- **Block 3**  
  - Conv3×3, 256 filters + ReLU  
  - Conv3×3, 256 filters + ReLU  
  - MaxPool 2×2

- **Fully Connected**  
  - FC 512 + ReLU  
  - FC 10 + Softmax

- **Optimization**  
  - Loss: Cross‑Entropy  
  - Optimizer: SGD/Adam  
  - Parallelization: Viết bằng Numpy → tối ưu với Numba (`@cuda.jit`)

Kiến trúc VGG‑like này giữ tinh thần “nhiều conv 3×3” của VGG gốc nhưng thu gọn cho phù hợp với kích thước ảnh 32×32 và dataset CIFAR‑10.

## **Dataset**

- **CIFAR-10**:
  - 60.000 ảnh màu 32×32 thuộc 10 lớp.

  - 50.000 ảnh train, 10.000 ảnh test.

  - Link: https://www.cs.toronto.edu/~kriz/cifar.html

## **Nội dung đã hoàn thành**

7 phiên bản bao gồm:
- Python thuần
- Numpy
- Song song hóa cơ bản
- Song song tối ưu hóa 1 - sử dụng stream
- Song song tối ưu hóa 2 - sử dụng shared memory (tối ưu từ phiên bản song song hóa cơ bản)
- Song song tối ưu hóa 3 - sử dụng shared memory + loop unrolling
- Song song tối ưu hóa 4 - sử dụng stream + shared memory + loop unrolling.

## **Cây thư mục**
```
📁 ./
├── .gitignore
├── README.md
├── requirements_python311.txt

├── 📁 Data/
│   ├── cifar-10-python.tar.gz
│   ├── 📁 log/
│   │   ├── details_log_cuda_ver4_10.csv
│   │   ├── details_log_cuda_ver4_50000_1.csv
│   │   ├── details_log_cuda_ver4_50000_op2_1.csv
│   │   ├── details_log_cuda_ver4_50000_op3_1.csv
│   │   ├── details_log_cuda_ver4_50000_op4_1.csv
│   │   ├── details_log_cuda_ver4_50000_op4_10.csv
│   │   ├── details_log_cuda_ver4_50000_op_1.csv
│   │   ├── details_log_cuda_ver4_op2_10.csv
│   │   ├── details_log_cuda_ver4_op3_10.csv
│   │   ├── details_log_cuda_ver4_op4_10.csv
│   │   ├── details_log_cuda_ver4_op_10.csv
│   │   ├── details_log_numpy.csv
│   │   ├── details_log_numpy_10.csv
│   │   ├── details_log_numpy_50000.csv
│   │   ├── details_log_numpy_50000_10.csv
│   │   ├── details_log_python.csv
│   │   ├── training_log_cuda_ver4_10.csv
│   │   ├── training_log_cuda_ver4_50000_1.csv
│   │   ├── training_log_cuda_ver4_50000_op2_1.csv
│   │   ├── training_log_cuda_ver4_50000_op3_1.csv
│   │   ├── training_log_cuda_ver4_50000_op4_1.csv
│   │   ├── training_log_cuda_ver4_50000_op4_10.csv
│   │   ├── training_log_cuda_ver4_50000_op_1.csv
│   │   ├── training_log_cuda_ver4_op2_10.csv
│   │   ├── training_log_cuda_ver4_op3_10.csv
│   │   ├── training_log_cuda_ver4_op4_10.csv
│   │   ├── training_log_cuda_ver4_op_10.csv
│   │   ├── training_log_numpy.csv
│   │   ├── training_log_numpy_10.csv
│   │   ├── training_log_numpy_50000.csv
│   │   ├── training_log_numpy_50000_10.csv
│   │   ├── training_log_python.csv

├── 📁 Src/
│   ├── func.py
│   ├── Image_Classification.ipynb
│   ├── Image_Classification_Numba_ver4.ipynb
│   ├── Image_Classification_Numba_ver4_op.ipynb
│   ├── Image_Classification_Numba_ver4_op_2.ipynb
│   ├── Image_Classification_Numba_ver4_op_3.ipynb
│   ├── Image_Classification_Numba_ver4_op_4.ipynb
│   ├── Image_Classification_Numpy.ipynb
```

## Hướng dẫn sử dụng trong Visual Studio Code
### 1. Môi trường Python
- Project này sử dụng **Python 3.11**.
- Cần bảo đảm Python 3.11 đã được cài và chọn đúng môi trường trong VSCode.

### 2. Cài đặt thư viện cần thiết
- Mở Terminal (trong VSCode hoặc ngoài đều được)
- Chạy lệnh sau để cài đặt các thư viện cần thiết từ file yêu cầu:

```bash
pip install -r requirements_python311.txt