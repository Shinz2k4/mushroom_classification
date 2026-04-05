# MushroomNet

Mô phỏng bài báo **"Mushroom image recognition and distance generation based on attention-mechanism model and genetic information"** — Wenbin Liao et al. (2022).

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Cấu trúc thư mục

```
dataset/raw/Mushroom/
    train/      # 80% ảnh (theo bài báo)
    valid/      # 10% ảnh
    test/       # 10% ảnh
models/         # file .pth được lưu tại đây sau khi train
src/            # toàn bộ source code
```

---

## 1. Train

### Train MushroomNet (Stage 2 + Stage 3, mặc định)

```bash
python src/main.py train
```

> Chạy đủ Stage 2 (30 epoch, fine-tune toàn mạng) rồi Stage 3 (30 epoch, chỉ train attention + head).  
> Model tốt nhất được lưu vào `models/mushroomnet.pth`.

---

### Chỉ chạy Stage 2

```bash
python src/main.py train --stage 2
```

### Chỉ chạy Stage 3

```bash
python src/main.py train --stage 3
```

> Yêu cầu `models/mushroomnet.pth` đã tồn tại từ Stage 2.

---

### Train ResNet50 (model so sánh)

```bash
python src/main.py train --model resnet50
```

> Chỉ chạy Stage 2. Model lưu vào `models/resnet50.pth`.

---

### Tùy chỉnh số epoch

```bash
python src/main.py train --epochs2 30 --epochs3 30
```

---

## 2. Đánh giá (Test)

### Đánh giá nhanh — chỉ loss và accuracy

```bash
python src/main.py test
```

### Đánh giá đầy đủ — Accuracy, Precision, F1, Recall + Confusion Matrix + ROC Curve

```bash
python src/main.py test --report
```

> In bảng kết quả từng loài (Acc/Prec/F1/Recall).  
> Lưu `models/confusion_matrix.png` và `models/roc_curve.png`.

---

### Đánh giá ResNet50

```bash
python src/main.py test --model resnet50 --report
```

---

## 3. Dự đoán ảnh mới

### Dự đoán 1 ảnh

```bash
python src/main.py predict --image path/to/mushroom.jpg
```

> In tên loài và confidence. Hiển thị Top-5 loài có xác suất cao nhất.

---

### Dự đoán + lưu ảnh Grad-CAM heat map

```bash
python src/main.py predict --image path/to/mushroom.jpg --gradcam 
```
<!-- python src/main.py predict --image D:\Codes\mang_noron\phan_biet_nam-btl_mang_noron\predict_folder\Hydnellum-peckii-650.jpg --gradcam -->

> Lưu file `models/gradcam_<tên_ảnh>.png` gồm 3 panel: ảnh gốc, heat map, overlay.

---

### Dự đoán toàn bộ ảnh trong 1 thư mục

```bash
python src/main.py predict --dir path/to/folder/
```

---

## 4. Genetic Distance (Section 2.5 bài báo)

### Train model genetic distance (mặc định: MSE-mean)

```bash
python src/main.py genetic
```

> Học cách map feature ảnh → không gian genetic distance.  
> Model lưu vào `models/mushroomnet_genetic.pth`.

---

### Các biến thể activation function (Section 2.5.3)

```bash
# MSE với reduction = sum
python src/main.py genetic --activation mse-sum

# MSE với reduction = mean  (mặc định)
python src/main.py genetic --activation mse-mean

# Softmax trước MSE
python src/main.py genetic --activation softmax
```

---

### Improved diagonal = -1 (Section 2.5.5)

```bash
python src/main.py genetic --diagonal-minus1
```

> Đặt các phần tử đường chéo của ma trận genetic distance target = -1 thay vì 0.

---

### Chỉ đánh giá (không train lại)

```bash
# Đánh giá bằng cosine distance (mặc định)
python src/main.py genetic --eval --metric cosine

# Đánh giá bằng Euclidean distance
python src/main.py genetic --eval --metric euclidean
```

---

## Kết quả đầu ra

| File | Nội dung |
|---|---|
| `models/mushroomnet.pth` | Trọng số MushroomNet tốt nhất |
| `models/resnet50.pth` | Trọng số ResNet50 tốt nhất |
| `models/mushroomnet_genetic.pth` | Trọng số model genetic distance |
| `models/training_history.png` | Biểu đồ Accuracy + Loss theo epoch |
| `models/confusion_matrix.png` | Confusion Matrix (khi dùng `--report`) |
| `models/roc_curve.png` | ROC Curve (khi dùng `--report`) |
| `models/gradcam_<tên>.png` | Grad-CAM heat map (khi dùng `--gradcam`) |
