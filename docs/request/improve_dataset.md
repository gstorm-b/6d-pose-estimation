Hãy kiểm tra dataset và giải quyết vấn đề object bị out khỏi box.
Mình đề xuất:
    - Chỉ load môi trường blender một lần để giảm thời giam generate.
    - Sau khi drop xong, ẩn hết tất cả object và hiện lần lượt để check visibility, rồi lại ẩn, để có thể đếm được số lượng object trong box, xem có object nào bị văng ra không.
    - Sau khi xong một scene thì xóa hết tất cả object và generate lại không cần phải load lại môi trường.
Bạn hãy kiểm tra xem đề xuất của mình có hợp lí không. Bạn đề xuất những phương án nào.

## Quyết định implement

- Giữ kiến trúc generator hiện tại: load Blender scene một lần, giữ bin/camera/light/STL template, mỗi attempt chỉ spawn object động rồi xóa object động sau khi xong.
- Không dùng cách ẩn/hiện từng object làm tiêu chí visibility chính, vì training cần visibility trong cảnh thật có occlusion. Generator tiếp tục raycast toàn scene để đếm instance visible và point visible.
- Kiểm tra out-of-bin bằng world-space bbox sau physics. Mặc định reject toàn bộ attempt nếu có bất kỳ object nào ra khỏi box.
- `--allow-out-of-bin-filtering` chỉ là legacy/debug mode. Dataset dùng để train phải dùng strict reject policy.
- Thêm CLI `scripts/validate_raw_dataset.py` để audit dataset sau generate và fail nếu dataset dùng legacy filtering, trừ khi chủ động truyền `--allow-legacy-filtering`.

## Kế hoạch noise / augmentation trước PointNet++

- Không làm bẩn raw Blender dataset. Raw data vẫn clean để dễ trace/debug ground truth.
- Thêm noise ở processed dataset hoặc online DataLoader augmentation.
- Implement trước data interface cho PointNet++ rồi mới train model, để tránh phải sửa lại training loop khi schema đổi.

Đã implement MVP:

- `scripts/prepare_pointnet2_semseg_dataset.py`: convert raw dataset sang processed PointNet++ semantic segmentation dataset.
- `src/data/depth_unprojection.py`: reconstruct full-scene point cloud từ `depth_m`.
- `src/data/pointnet2_processing.py`: balanced sampling object/background và lưu `points`, `features`, `semantic_labels`, `instance_labels`, `point_pixels`.
- `src/data/augmentation.py`: noise hooks gồm xyz jitter, depth noise, dropout, outlier, normal jitter.
- `src/data/pointnet2_dataset.py`: PyTorch dataset wrapper cho training sau này.
- `configs/train/pointnet2_semseg_k41144.yaml`: config mặc định cho conversion/augmentation.

## Bước tiếp theo đã triển khai

- Đã validate raw `synthetic-data/K41144`: 77/77 samples OK, không out-of-bin.
- Đã phát hiện processed dataset full hiện có tại `processed-data/pointnet2_semseg_k41144`.
- Stats processed dataset: 77 samples, split train/val/test = 62/8/7, mỗi sample 16384 points, object/background = 10650/5734.
- Đã thêm PointNet++ semantic segmentation MVP:
  - `src/models/pointnet2_ops.py`
  - `src/models/pointnet2_semseg.py`
  - `scripts/train_pointnet2_semseg.py`
  - `src/training/config.py`
  - `src/training/metrics.py`
  - `src/training/checkpoint.py`
  - `src/training/seed.py`

Ghi chú: môi trường hiện tại chưa cài PyTorch, nên chưa chạy runtime training được. Sau khi cài `requirements-train.txt`, chạy smoke training:

```powershell
python .\scripts\train_pointnet2_semseg.py --config .\configs\train\pointnet2_semseg_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144 --epochs 2 --batch-size 2 --device cpu
```

## Cập nhật môi trường train

- Đã thay PyTorch CPU bằng PyTorch GPU/CUDA: `torch 2.12.0+cu126`.
- Đã cài thêm `pyyaml`, `tqdm`, `scikit-learn`; `matplotlib` đã có sẵn.
- CUDA khả dụng: `torch.cuda.is_available() == True`.
- GPU: `NVIDIA GeForce MX150`, compute capability `(6, 1)`.
- PyTorch CUDA arch list có `sm_61`, nên wheel CUDA 12.6 hỗ trợ GPU này.
- Vì GPU chỉ có 2GB VRAM, nên nên train với `batch_size=1` và dataset/debug config nhỏ trước. Full 16384 points có thể OOM.
- Đã smoke-test DataLoader và model forward pass.
- Đã chạy smoke training CPU 1 epoch trên `processed-data/pointnet2_semseg_k41144_smoke_config`.
- Đã chạy smoke training GPU 1 epoch trên cùng smoke dataset.
- Experiment CPU smoke:

```text
experiments/pointnet2_semseg_k41144_20260530_151203/
```

- Experiment GPU smoke:

```text
experiments/pointnet2_semseg_k41144_20260530_153222/
```

Kết quả GPU smoke:

```text
train_loss=0.7583
val_loss=0.6951
val_miou=0.1750
val_object_recall=0.0000
```

Metric này chỉ xác nhận pipeline chạy end-to-end, chưa dùng để đánh giá chất lượng model.

## Phase 1 PointNet++ Debug Dataset

Đã tạo debug processed dataset:

```text
processed-data/pointnet2_semseg_k41144_debug4096
```

Kết quả kiểm tra:

```text
sample_count: 77
train/val/test: 62/8/7
points per sample: 4096
object/background per sample: 2662/1434
features: normal_camera, shape (4096, 3)
labels: checked train/val/test samples all contain both label 0 and 1
```

## Phase 2 PointNet++ Smoke Test

Đã chạy smoke test trên debug4096:

```text
DataLoader train length: 62
model_input shape: (4096, 6)
semantic_labels shape: (4096,)
labels present: 0 and 1
CUDA forward/backward: pass
logits shape: (1, 4096, 2)
device: cuda:0
```

Phase tiếp theo: thêm overfit flags cho training script, rồi chạy one-sample overfit.

## Phase 3 One-Sample Overfit

Đã thêm flags cho training script:

```text
--limit-train-samples
--limit-val-samples
--disable-augment
--overfit
--dropout
--learning-rate
```

Đã chạy one-sample overfit:

```text
experiment: experiments/pointnet2_semseg_k41144_20260606_153430
dataset: processed-data/pointnet2_semseg_k41144_debug4096
epochs: 50
batch_size: 1
device: cuda
overfit: true
limit_train_samples: 1
limit_val_samples: 1
augmentation: disabled
dropout: 0.0
learning_rate: 0.001
```

Final train-mode metrics:

```text
train_loss=0.0604
train_mean_iou=0.9588
train_object_iou=0.9706
train_object_recall=0.9790
```

Final eval-mode metrics trên cùng sample:

```text
val_loss=0.3633
val_mean_iou=0.7002
val_object_iou=0.7429
val_object_recall=0.7652
```

Kết luận:

- One-sample train-mode overfit pass, chứng minh data/loss/backward path hoạt động.
- Eval-mode vẫn yếu hơn đáng kể, có thể do BatchNorm running stats với `batch_size=1`.
- Phase tiếp theo là two-sample overfit và theo dõi eval-mode gap. Nếu gap vẫn lớn, cần đổi/tune normalization.

## Phase 4 Two-Sample Overfit

Đã chạy two-sample overfit:

```text
experiment: experiments/pointnet2_semseg_k41144_20260606_153959
dataset: processed-data/pointnet2_semseg_k41144_debug4096
epochs: 80
batch_size: 1
device: cuda
overfit: true
limit_train_samples: 2
limit_val_samples: 2
augmentation: disabled
dropout: 0.0
learning_rate: 0.001
```

Best eval-mode epoch:

```text
epoch=78
train_loss=0.0115
train_mean_iou=0.9939
train_object_iou=0.9957
train_object_recall=0.9976
val_loss=0.0121
val_mean_iou=0.9933
val_object_iou=0.9953
val_object_recall=0.9970
```

Final epoch:

```text
epoch=80
train_loss=0.0098
train_mean_iou=0.9939
train_object_iou=0.9957
train_object_recall=0.9966
val_loss=0.0350
val_mean_iou=0.9663
val_object_iou=0.9758
val_object_recall=0.9771
```

Kết luận:

- Phase 4 pass rất tốt.
- Eval-mode gap ở Phase 3 không còn là blocker rõ ràng khi overfit 2 samples.
- Phase tiếp theo là tạo prediction preview tool để nhìn trực quan ground truth/prediction/error trước khi train full split.

## Phase 5 Prediction Preview Tool

Đã thêm preview script:

```text
scripts/preview_pointnet2_semseg_predictions.py
```

Đã chạy preview export:

```text
checkpoint: experiments/pointnet2_semseg_k41144_20260606_153959/checkpoints/best.pt
data: processed-data/pointnet2_semseg_k41144_debug4096
split: train
max_samples: 2
device: cuda
output: experiments/pointnet2_semseg_k41144_20260606_153959/previews/train
```

Generated files:

```text
preview_summary.json
sample_000000_gt.ply
sample_000000_pred.ply
sample_000000_error.ply
sample_000001_gt.ply
sample_000001_pred.ply
sample_000001_error.ply
```

Preview metrics:

```text
overall_accuracy=0.9972
mean_iou=0.9938
object_iou=0.9957
object_recall=0.9981
confusion_matrix=[[2855, 13], [10, 5314]]
```

PLY check:

```text
format: ASCII PLY
vertices per file: 4096
```

Ghi chú: export tool đã hoạt động. Vẫn nên mở PLY bằng Open3D/MeshLab để kiểm tra trực quan màu ground truth/prediction/error trước khi train full split.

## Phase 6 Debug Dataset Training

Đã train full debug4096 split:

```text
experiment: experiments/pointnet2_semseg_k41144_20260606_155444
dataset: processed-data/pointnet2_semseg_k41144_debug4096
epochs: 20
batch_size: 1
device: cuda
train/val split: 62/8
runtime: about 16.2 minutes on NVIDIA GeForce MX150
```

Best validation epoch:

```text
epoch=19
train_loss=0.0054
train_mean_iou=0.9959
train_object_iou=0.9971
train_object_recall=0.9979
val_loss=0.0066
val_mean_iou=0.9937
val_object_iou=0.9956
val_object_recall=0.9961
```

Final epoch:

```text
epoch=20
train_loss=0.0067
train_mean_iou=0.9949
train_object_iou=0.9964
train_object_recall=0.9973
val_loss=0.0180
val_mean_iou=0.9865
val_object_iou=0.9905
val_object_recall=0.9911
```

Đã export validation previews:

```text
checkpoint: experiments/pointnet2_semseg_k41144_20260606_155444/checkpoints/best.pt
output: experiments/pointnet2_semseg_k41144_20260606_155444/previews/val
sample_count: 8
files: 8 gt PLY + 8 pred PLY + 8 error PLY + preview_summary.json
```

Preview overall metrics:

```text
overall_accuracy=0.9969
mean_iou=0.9931
object_iou=0.9952
object_precision=0.9997
object_recall=0.9954
confusion_matrix=[[11466, 6], [97, 21199]]
```

Kết luận:

- Phase 6 pass rất tốt trên debug4096.
- Bước tiếp theo là Phase 7: profile memory/speed để quyết định có train được 8192 hoặc 16384 points trên MX150 2GB không.

## Phase 7 Memory/Speed Profiling

Đã thêm profiler:

```text
scripts/profile_pointnet2_memory.py
```

Đã tạo thêm processed dataset để profile:

```text
processed-data/pointnet2_semseg_k41144_debug8192
sample_count: 77
train/val/test: 62/8/7
points per sample: 8192
object/background per sample: 5325/2867
```

Profile output:

```text
experiments/pointnet2_profile_20260606/
experiments/pointnet2_profile_20260606/profile_summary.json
```

GPU:

```text
NVIDIA GeForce MX150
total memory: 2047.875 MiB
```

Profile matrix:

```text
4096 points, sa_npoints=[1024,256,64], sa_nsamples=[32,32,32]
  status=ok
  mean_seconds_per_batch=1.0731
  max_allocated_mib=130.1
  max_reserved_mib=148.0

8192 points, sa_npoints=[1024,256,64], sa_nsamples=[32,32,32]
  status=ok
  mean_seconds_per_batch=0.9815
  max_allocated_mib=153.5
  max_reserved_mib=170.0

16384 points, sa_npoints=[1024,256,64], sa_nsamples=[32,32,32]
  status=ok
  mean_seconds_per_batch=2.1114
  max_allocated_mib=199.9
  max_reserved_mib=228.0

16384 points, sa_npoints=[4096,1024,256], sa_nsamples=[32,32,32]
  status=ok
  mean_seconds_per_batch=5.7413
  max_allocated_mib=669.3
  max_reserved_mib=804.0
```

Kết luận:

- MX150 fit được tất cả config đã profile với `batch_size=1`.
- Config hiện tại `sa_npoints=[1024,256,64]` an toàn cho baseline 16384 points.
- Config lớn `sa_npoints=[4096,1024,256]` cũng fit nhưng chậm hơn nhiều, chưa nên dùng làm baseline đầu tiên.
- Phase tiếp theo là Phase 8: chạy full synthetic baseline với `processed-data/pointnet2_semseg_k41144`, `batch_size=1`, `device=cuda`.

## Tạo noise cho dataset cho gần với thực tế:
- Hiện tại dataset pointcloud được generate khác clean, trong thực tế:
    - Nhiễu ánh sáng sẽ làm mất đi những cụm point cloud tại khu vực nhiễu.
    - Khu vực viền giữa hai object thường không có sự khác biệt rõ ràng, mà khoảng cách space giữa các điểm point cloud ở đây sẽ khá sát nhau.
Bạn có giải pháp nào để tạo dataset sát với thực tế hoặc giải pháp nào trong quá trình train có thể giải quyết vấn đề này không.
