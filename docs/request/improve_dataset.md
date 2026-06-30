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
- Bước tiếp theo từ Phase 7 là Phase 8: chạy full synthetic baseline với `processed-data/pointnet2_semseg_k41144`, `batch_size=1`, `device=cuda`.

## Phase 8 Full Synthetic Baseline

Đã train full processed dataset 16384 points:

```text
experiment: experiments/pointnet2_semseg_k41144_20260606_180212
dataset: processed-data/pointnet2_semseg_k41144
epochs: 20
batch_size: 1
device: cuda
runtime: about 21.6 minutes on NVIDIA GeForce MX150
```

Best validation epoch:

```text
epoch=19
val_loss=0.00499
val_mean_iou=0.9961
val_object_iou=0.9973
val_object_recall=0.9984
```

Final epoch:

```text
epoch=20
val_loss=0.00518
val_mean_iou=0.9948
val_object_iou=0.9964
val_object_recall=0.9969
```

Đã export validation previews:

```text
checkpoint: experiments/pointnet2_semseg_k41144_20260606_180212/checkpoints/best.pt
output: experiments/pointnet2_semseg_k41144_20260606_180212/previews/val
sample_count: 8
files: 8 gt PLY + 8 pred PLY + 8 error PLY + preview_summary.json
```

Preview overall metrics:

```text
overall_accuracy=0.9982
mean_iou=0.9960
object_iou=0.9972
object_precision=0.9990
object_recall=0.9982
confusion_matrix=[[45789,83],[155,85045]]
```

Kết luận:

- Phase 8 pass rất tốt trên full synthetic validation split.
- Checkpoint `best.pt` hiện là baseline semantic segmentation chính đầu tiên.
- Vẫn cần mở PLY previews để kiểm tra trực quan trước khi sign-off chất lượng.
- Phase tiếp theo là Phase 9: thêm test evaluation script và chạy held-out test split.

## PointNet++ PyTorch3D Ops Optimization

Sau khi thấy GPU usage chỉ khoảng 20% và training còn chậm, đã kiểm tra PyTorch3D trong environment hiện tại:

```text
torch=2.12.0+cu126
pytorch3d=0.7.9
pytorch3d._C available
GPU=NVIDIA GeForce MX150
```

Ops có sẵn và chạy CUDA được:

```text
sample_farthest_points
knn_points
knn_gather
ball_query
```

Đã implement backend config:

```text
src/models/pointnet2_ops.py
  pure_torch backend
  pytorch3d backend
  fallback builder

src/models/pointnet2_semseg.py
  model.ops_backend support

configs/train/pointnet2_semseg_k41144.yaml
  ops_backend: pytorch3d
```

Benchmark:

```text
current_fps_1024: 0.4284 s
pytorch3d_fps_1024: 0.0211 s
```

Profiler end-to-end trên full 16384-point dataset, batch size 1:

```text
pure_torch:
  mean_seconds_per_batch=0.8340
  max_allocated_mib=199.9
  max_reserved_mib=228.0

pytorch3d:
  mean_seconds_per_batch=0.1143
  max_allocated_mib=135.7
  max_reserved_mib=158.0
```

Batch-size probe với PyTorch3D backend:

```text
batch_size=2:
  mean_seconds_per_batch=0.1872
  max_reserved_mib=288.0

batch_size=4:
  mean_seconds_per_batch=0.4654
  max_reserved_mib=572.0
```

Validation:

```text
CUDA forward/backward with pytorch3d backend: pass
train smoke with config backend=pytorch3d: pass
```

Kết luận:

- Nên dùng PyTorch3D backend trước, chưa build custom extension.
- PyTorch3D cho speedup khoảng 7x trên profiled train batch và dùng ít GPU memory hơn.
- `batch_size=2` hoặc `batch_size=4` hiện đều fit trên MX150 với config đã profile.
- Giữ `pure_torch` làm fallback/debug backend.
- Nếu vẫn cần nhanh hơn, bước tiếp theo nên là tune batch size, DataLoader workers, pinned memory, và giảm CPU sync metrics trước khi viết fused/custom CUDA kernel.

## Phase 9 Held-Out Test Evaluation

Đã thêm evaluation script:

```text
scripts/eval_pointnet2_semseg.py
```

Đã chạy held-out synthetic test split với PyTorch3D backend:

```text
checkpoint: experiments/pointnet2_semseg_k41144_20260606_180212/checkpoints/best.pt
data: processed-data/pointnet2_semseg_k41144
split: test
output: experiments/pointnet2_semseg_k41144_20260606_180212/test_metrics.json
sample_count: 7
device: cuda
ops_backend: pytorch3d
seed: 7
```

Test metrics:

```text
loss=0.0075
overall_accuracy=0.9972
mean_iou=0.9939
object_iou=0.9957
object_precision=0.9991
object_recall=0.9966
confusion_matrix=[[40074,64],[254,74296]]
```

Đã export test previews:

```text
output: experiments/pointnet2_semseg_k41144_20260606_180212/previews/test
sample_count: 7
files: 7 gt PLY + 7 pred PLY + 7 error PLY + preview_summary.json
```

Kết luận:

- Phase 9 pass: test metrics gần validation metrics.
- Không có dấu hiệu collapse toàn background hoặc toàn object.
- Confusion matrix đã được lưu trong `test_metrics.json`.
- Vẫn cần kiểm tra trực quan PLY previews trước khi sign-off semantic segmentation.
- Phase tiếp theo là Phase 10: inference script/API.

## Phase 10 Inference API

Đã thêm reusable inference API và CLI:

```text
src/inference/pointnet2_semseg_infer.py
scripts/infer_pointnet2_semseg.py
```

Processed sample inference:

```powershell
python .\scripts\infer_pointnet2_semseg.py --checkpoint .\experiments\pointnet2_semseg_k41144_20260606_180212\checkpoints\best.pt --sample .\processed-data\pointnet2_semseg_k41144\test\sample_000011.npz --sample-type processed --out .\experiments\pointnet2_semseg_k41144_20260606_180212\inference\processed_sample_000011 --device cuda --ops-backend pytorch3d --save-ply
```

Processed result:

```text
points=16384
predicted_object_points=10632
ground_truth_object_points=10650
elapsed_seconds=0.4877
synthetic_debug_accuracy=0.9976
```

Raw sample inference:

```powershell
python .\scripts\infer_pointnet2_semseg.py --checkpoint .\experiments\pointnet2_semseg_k41144_20260606_180212\checkpoints\best.pt --sample .\synthetic-data\K41144\sample_000011 --sample-type raw --out .\experiments\pointnet2_semseg_k41144_20260606_180212\inference\raw_sample_000011 --device cuda --ops-backend pytorch3d --num-points 16384 --save-ply
```

Raw result:

```text
points=16384
predicted_object_points=6403
ground_truth_object_points=6440
elapsed_seconds=0.5222
synthetic_debug_accuracy=0.9964
```

Output files:

```text
prediction.npz
summary.json
prediction.ply
```

Kết luận:

- Phase 10 pass: inference chạy độc lập khỏi training loop.
- Output `points_camera`, `point_pixels`, `logits`, `probabilities`, `predicted_labels` align theo row.
- Processed inference dùng `.npz` đã convert.
- Raw inference reconstruct point cloud từ `depth_m + camera_intrinsics`, sample không dùng label, dùng `normal_camera` nếu checkpoint cần normals.
- Synthetic `instance_mask` chỉ được copy vào output để debug, không dùng cho sampling/inference.

## Tạo noise cho dataset cho gần với thực tế:
- Hiện tại dataset pointcloud được generate khác clean, trong thực tế:
    - Nhiễu ánh sáng sẽ làm mất đi những cụm point cloud tại khu vực nhiễu.
    - Khu vực viền giữa hai object thường không có sự khác biệt rõ ràng, mà khoảng cách space giữa các điểm point cloud ở đây sẽ khá sát nhau.
Bạn có giải pháp nào để tạo dataset sát với thực tế hoặc giải pháp nào trong quá trình train có thể giải quyết vấn đề này không.

### Cập nhật 2026-07-01 - real-depth gap từ `real-data/scene_001.pcd`

File real tham chiếu hiện có là `real-data/scene_001.pcd` (không thấy folder tên
`real scene`). Đây là point cloud binary PCD với fields `x y z rgb`, 459708
points, `HEIGHT 1`. Tọa độ có scale dạng mm: bbox khoảng
`x=-126..169`, `y=-161..211`, `z=-725..-595`; khi đưa vào real-eval phải scale
`0.001` sang mét.

Thống kê sau khi scale sang mét:

- extent scene: khoảng `0.296 x 0.372 x 0.130 m`.
- nearest-neighbor XY trên sample 80k: p50/p90/p99 khoảng `0.59 / 0.96 / 1.12 mm`.
- local depth gap giữa 8 hàng xóm XY: p50/p90/p95/p99 khoảng
  `0.85 / 28.27 / 56.89 / 96.55 mm`.
- khoảng `19.56%` điểm có local depth gap `> 5 mm`, `17.80%` điểm `> 10 mm`.

Nhận định senior-engineer:

- Synthetic raycast hiện quá sạch ở biên object và bề mặt specular/đen. Gaussian
  jitter nhẹ không đủ mô phỏng việc camera thật mất depth, fatten edge, hoặc làm
  các object sát nhau dính thành một cụm.
- Không nên ghi noise trực tiếp vào raw `synthetic-data/`; raw vẫn cần clean để
  giữ GT/mask/pose có thể audit. Noise nên nằm trong train-time augmentation hoặc
  một processed experiment riêng có manifest rõ ràng.
- Repo đã có augmentation đúng hướng trong `src/data/augmentation.py`:
  range-squared depth noise, depth quantization, grazing dropout, edge dropout,
  blob/specular dropout. Cần bật nó cho K41144 Wave2 và đo clean regression trước
  khi promote model.

Thay đổi đã thêm:

- `configs/train/pointnet2_instance_k41144_wave2_sim2real.yaml`: config instance
  segmentation K41144 Wave2 bật structured-light sim-to-real noise, root trỏ
  `processed-data/pointnet2_semseg_k41144_wave2`, batch size 4.
- `scripts/generate_synthetic_blender.py`: thêm post-raycast sensor artifact
  simulation cho raw generation. Profile `none` giữ behavior cũ; profile
  `structured_light` / `tof` bật depth quantization, range noise, random/edge/blob
  dropout, edge smear, flying pixels, và bridge artifact. Output `sensor_data.npz`
  có thêm `sensor_artifact_mask`; `metadata.json` có `sensor_noise_summary`.
- `scripts/dataset_gui.py`: thêm group `Depth Sensor Noise` để chỉnh/export/import
  toàn bộ tham số generator sensor-noise từ GUI.

Backlog đề xuất để data giống real hơn:

1. Train lại K41144 Wave2 với config sim-to-real mới, so với clean checkpoint trên
   cùng synthetic test gate và trên real set đã label. Không promote nếu clean
   ADD_0.1d/instance recall tụt quá ngưỡng.
2. Convert `real-data/scene_001.pcd` thành real-eval frame chuẩn nếu có intrinsics
   hoặc export thêm depth frame gốc từ camera. PCD hiện không đủ để sinh mask 2D
   hoặc chạy full depth pipeline như `depth.npy + intrinsics.json`.
3. Calibrate noise bằng real frames: match invalid/dropout fraction, depth edge
   gap distribution, quantization, và blob hole size thay vì dùng guess cố định.
4. Mở rộng generator scene composition: ưu tiên dense piles/touching objects,
   nhiều object liền kề, occlusion cao, fill level giống bin thật; giảm các scene
   quá "đẹp" hoặc quá tách biệt.
5. Calibrate `edge_smear_prob`, `flying_pixel_prob`, `bridge_prob`,
   `edge_dropout_prob`, `blob_dropout_count`, và `depth_quadratic_noise` theo
   real frames thay vì dùng default. Mục tiêu là match distribution của
   `sensor_artifact_mask`, object-point retention, và local depth-gap với real PCD.
