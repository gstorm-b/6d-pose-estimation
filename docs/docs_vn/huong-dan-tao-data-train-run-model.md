# Hướng Dẫn Tạo Data, Train Và Chạy Model

Ngày cập nhật: 2026-06-30

Trạng thái: pipeline research đã hoàn thành đến pose Phase 23; đã bổ sung dataset lớn (Wave 2) và backend phục vụ model (Wave 1-3: registry/bundle, library pipeline, HTTP service, instance-seg tool, eval harness). Backbone pose chính là PointNet++ theo hướng PS6D-style keypoint/center voting. Refinement bằng ICP và learned translation refiner đã có, nhưng phải báo cáo tách biệt với raw pose.

**Đọc section "Cập Nhật 2026-06-30" và "Cập Nhật 2026-06-16" ngay dưới đây trước** — chúng chứa reference checkpoint/metric MỚI, cách gộp dataset lớn, cách đóng gói + chạy backend, và các kết quả âm KHÔNG nên lặp lại. Phần còn lại (mục 1-21) vẫn đúng cho pipeline cơ bản.

Tài liệu này là runbook tiếng Việt để engineer có thể dựng lại toàn bộ pipeline:

```text
Blender synthetic data
  -> processed PointNet++ semantic/instance dataset
  -> train instance segmentation
  -> export pose crops từ GT hoặc predicted instances
  -> train pose keypoint/center voting
  -> evaluate raw pose
  -> optional refined pose
  -> optional learned translation refiner
  -> full pose evaluation suite
```

## Cập Nhật 2026-06-30 (K41144 Wave 2 từ k41144_1..k41144_8)

K41144 đã được chạy theo quy trình Wave 2 từ các raw roots:

```text
synthetic-data/k41144_1 ... synthetic-data/k41144_8
```

Raw validation:

```text
k41144_1: 200/200 OK, 30 visible objects
k41144_2: 0/200 OK do SPARSE, nhưng không corrupt; 10 visible objects, no out-of-bin
k41144_3: 200/200 OK, 20 visible objects
k41144_4: 200/200 OK, 3 visible objects
k41144_5: 107/107 OK, 50 visible objects
k41144_6: 200/200 OK, 8 visible objects
k41144_7: 200/200 OK, 8 visible objects
k41144_8: 87/88 OK; sample_000087 thiếu toàn bộ file raw
```

Converter đã có thêm flag opt-in:

```text
--skip-invalid-samples
```

Flag này chỉ bỏ qua sample folder thiếu file bắt buộc khi merge multi-root trusted data; default vẫn strict. Lệnh đã dùng:

```powershell
python .\scripts\prepare_pointnet2_semseg_dataset.py --raw .\synthetic-data\k41144_1 .\synthetic-data\k41144_2 .\synthetic-data\k41144_3 .\synthetic-data\k41144_4 .\synthetic-data\k41144_5 .\synthetic-data\k41144_6 .\synthetic-data\k41144_7 .\synthetic-data\k41144_8 --out .\processed-data\pointnet2_semseg_k41144_wave2 --num-points 16384 --use-normals --object-fraction-target 0.65 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 7 --skip-raw-validation --skip-invalid-samples
```

Processed dataset:

```text
processed-data/pointnet2_semseg_k41144_wave2
samples: 1394
split train/val/test: 1115/139/140
visible_instances min/median/max: 3/10/50
skipped: synthetic-data/k41144_8/sample_000087
```

Instance checkpoint tạm thời:

```text
experiments/pointnet2_instance_k41144_wave2_20260630_231537/checkpoints/best.pt
training: 3 epochs, batch_size=4, device=cuda, num_workers=0
test semantic object_iou: 0.9932
test instance_precision: 0.8676
test instance_recall: 0.8054
test instance_mean_iou: 0.8600
```

Pose crop exports:

```text
GT train: experiments/wave2_pose_crops_k41144_gt_train, 18579 crops
GT val:   experiments/wave2_pose_crops_k41144_gt_val, 2251 crops
GT test:  experiments/wave2_pose_crops_k41144_gt_test, 2495 crops
Pred test from new instance: experiments/wave2_pose_crops_k41144_pred_test, 2181 crops
Pred matched_gt_iou mean: 0.7182
Pred matched_gt_iou >= 0.5: 1774/2181
```

Pose checkpoint:

```text
experiments/pointnet2_pose_k41144_20260630_233852/checkpoints/best_add.pt
source: fine-tuned 1 epoch from experiments/pointnet2_pose_k41144_20260608_023605/checkpoints/best_add.pt
```

GT-crop pose comparison on Wave 2 test:

```text
old checkpoint:       ADD 4.406 mm, translation 4.434 mm, ADD_0.1d 0.973
fine-tuned checkpoint ADD 3.099 mm, translation 3.199 mm, ADD_0.1d 0.961
```

Predicted-crop pose with fine-tuned checkpoint:

```text
matched_gt_iou >= 0.5:
  ADD 6.096 mm, translation 5.730 mm, ADD_0.1d 0.849
all predicted:
  ADD 10.817 mm, translation 11.280 mm, ADD_0.1d 0.732
```

Interpretation:

- Fine-tuning improves GT-crop ADD/translation substantially, but slightly lowers GT-crop ADD_0.1d versus the old checkpoint.
- Full predicted path is now limited mainly by instance crop quality, especially the dense 50-object scenes in `k41144_5`.
- Treat the 2026-06-30 checkpoints as **provisional**, not production-promoted, until the backlog below is complete.

Backlog:

1. Run longer instance training/sign-off (baseline target: original 50 epochs or early stop) and tune DBSCAN/min-cluster settings on Wave 2.
2. Run longer pose fine-tune and keep both selection criteria visible: best ADD and ADD_0.1d.
3. Run optional hybrid refinement on Wave 2 GT/predicted cases and report raw/refined separately.
4. Run Phase 21 K41144 predicted-crop sweep on `pointnet2_semseg_k41144_wave2`.
5. Package K41144 bundle v2 only after predicted-crop/full-pipeline metrics improve stably.

## Cập Nhật 2026-06-16 (Backend + Wave 2/3 — đọc trước)

Runbook gốc (mục 1-21) vẫn đúng cho pipeline cơ bản. Những điểm thay đổi quan trọng kể từ 2026-06-08:

### A. Gộp dataset lớn bằng multi-root (Wave 2)

`prepare_pointnet2_semseg_dataset.py --raw` giờ nhận NHIỀU root; sample được prefix theo root (vd `bending_pipe_10__sample_000000`). Đây là cách tạo dataset lớn từ nhiều lần generate:

```powershell
python .\scripts\prepare_pointnet2_semseg_dataset.py --raw .\synthetic-data\bending_pipe_4 .\synthetic-data\bending_pipe_5 .\synthetic-data\bending_pipe_6 .\synthetic-data\bending_pipe_18 --out .\processed-data\pointnet2_semseg_bending_pipe_wave2 --num-points 16384 --use-normals --skip-raw-validation
```

`SPARSE` = ít object visible, KHÔNG phải corrupt → dùng `--skip-raw-validation` khi đã tin data. Khi export pose crop từ dataset gộp, truyền `--raw-root .\synthetic-data` (root cha) để bridge tự resolve tên prefix.

### B. Reference checkpoint + metric MỚI (dùng cái này cho bending_pipe)

```text
Instance bending_pipe (Wave 2):
  experiments/pointnet2_instance_bending_pipe_wave2_20260614_154448/checkpoints/best.pt
  test object_iou 0.9997, instance_recall 0.967, split_count 1.78
Pose bending_pipe (Wave 2, GT crops):
  experiments/pointnet2_pose_bending_pipe_20260614_191629/checkpoints/best_add.pt
  GT-crop test (n=122): ADD 4.57 mm, translation 4.61 mm, ADD_0.1d 0.960
  (vượt xa reference cũ 9.9 mm / 0.833; driver chính là dataset lớn)
K41144: vẫn dùng checkpoint 2026-06-08 cho tới khi có data K41144 lớn → lặp Wave 2 cho K41144.
```

Two-stage clustering (merge/split) đã thử trên dense bending_pipe → KHÔNG cải thiện (merge lỏng gộp nhầm scene dày, merge chặt gần no-op) → **tắt mặc định**, giữ code config-gated.

### C. Đóng gói + phục vụ model (backend Wave 1-3)

Sau khi có instance + pose checkpoint, đóng gói thành bundle cho registry rồi phục vụ. Bundle tự chứa STL + model points/keypoints + diameter + symmetry + tham số inference.

```powershell
python .\scripts\package_model_bundle.py --sku bending_pipe --version v2 --instance-checkpoint .\experiments\pointnet2_instance_bending_pipe_wave2_20260614_154448\checkpoints\best.pt --pose-checkpoint .\experiments\pointnet2_pose_bending_pipe_20260614_191629\checkpoints\best_add.pt --object-probability-threshold 0.5 --dbscan-eps-m 0.006 --dbscan-min-samples 8 --min-cluster-points 32 --num-points 16384 --overwrite
```

- **Chạy end-to-end KHÔNG cần export crop thủ công**: dùng library `src/inference/pose_pipeline.py` (scene point cloud / depth → ranked `object_to_camera`) hoặc HTTP service `scripts/run_backend_service.py`. Hướng dẫn dùng đầy đủ: **docs/usage-guide.md**. API HTTP: **docs/backend-api.md**.
- **Instance segmentation độc lập**: `scripts/run_instance_segmentation.py` (từ bundle `--sku` hoặc checkpoint thô `--checkpoint`; input depth/points; output labels + clusters + PLY màu + mask 2D). Xem trực quan bằng app `scripts/view_instance_segmentation.py` (PySide6+VTK): **Run segmentation** (tô màu theo instance, click isolate, centroid, tinh chỉnh clustering live) **và Run pose** (overlay model đã pose + trục pose lên cảnh, xếp theo confidence, click isolate — kiểm tra model có "khớp" lên điểm quan sát không).
- **Confidence + chọn pick**: confidence giờ có term `model_fit` (chamfer model-đã-pose → crop) — pose sai bị hạ rank mạnh; **top-1 pick success ~1.0** trên pile dày. Option `max_model_fit` để gate. Chi tiết: **docs/dense-pile-pose-gap.md**.
- **Eval trên folder frame (synthetic/real)**: `scripts/prepare_real_eval_set.py` (export-synthetic / evaluate / label; báo ADD/ADD_0.1d/recall/top-K pick). Hướng dẫn capture real: **docs/real-data-capture-guide.md**.
- **Sim-to-real noise** (P8): bật bằng `configs/train/pointnet2_instance_bending_pipe_sim2real.yaml` (structured-light noise: along-ray range², quantization, grazing/edge/blob dropout — mặc định tắt ở config gốc).

### D. KẾT QUẢ ÂM đã kiểm chứng — KHÔNG lặp lại

- **Train pose voting FROM-SCRATCH trên predicted crops → TỆ HƠN GT-trained** (dense-pile per-instance precision 0.42 vs 0.58). Predicted crops có nhiều crop split/partial pose mơ hồ → tín hiệu train nhiễu. Giữ GT-trained + `model_fit`. Nếu muốn nâng per-instance precision: thử **finetune từ GT model** (`--resume`) hoặc **mix GT+predicted**, KHÔNG from-scratch. (docs/dense-pile-pose-gap.md)
- Gap predicted-path trên pile dày KHÔNG do contamination (96.9% cluster pure) cũng KHÔNG do symmetry thiếu (bending_pipe thật sự bất đối xứng) — mà do train/inference crop-distribution mismatch; đã mitigate bằng `model_fit` confidence.

### E. Training gotchas (GPU yếu / Windows)

- stdout của train script bị **block-buffered** khi ghi ra file → dùng `python -u` hoặc đọc `experiments/<run>/metrics.json` (ghi per-epoch) để xem tiến độ live.
- `--num-workers 0` an toàn trên Windows; `>0` có thể chậm spawn và không giúp nhiều khi GPU là nút thắt.
- MX150 chỉ 2GB VRAM; nếu chạy đồng thời Blender generation thì ~8 phút/epoch (cạnh tranh GPU). Kill đúng tiến trình train bằng match command-line (`train_pointnet2_pose` / `train_pointnet2_instance_seg`), tránh kill Blender.
- Khi `--resume`, training bắt đầu ở `epoch_checkpoint + 1`, nên `--epochs N` phải lớn hơn epoch của checkpoint thì mới train tiếp.
- `best_add.pt` luôn là checkpoint theo validation ADD tốt nhất → early-stop khi val plateau vẫn lấy được checkpoint tốt.

---

## 1. Quy Ước Quan Trọng

- Mỗi lần sản xuất thực tế chỉ picking một mẫu mã, nên dataset K41144 và bending_pipe được tạo, train, test riêng.
- Không trộn K41144 và bending_pipe vào cùng một dataset nếu chưa có phase multi-class rõ ràng.
- K41144 dùng `object-model/K41144.stl` với `model_scale=1.0`.
- bending_pipe dùng `object-model/bending_pipe.stl` với `model_scale=0.001`.
- Raw dataset trong `synthetic-data/` nên xem là immutable. Khi generate lại, tạo folder mới.
- Pose crop dùng frame camera theo metadata Blender. Script pose crop sẽ flip trục `z` từ depth positive-forward sang metadata camera frame.
- K41144 metric dùng symmetry `{I, R_y(pi)}`.
- bending_pipe metric dùng symmetry `{I}`.
- Luôn report riêng:
  - GT-crop pose
  - predicted-crop pose với `matched_gt_iou >= 0.5`
  - all predicted-crop pose
  - raw pose
  - refined pose nếu bật refinement

Tên folder nên theo pattern:

```text
synthetic-data/<object>_<run_name>
processed-data/pointnet2_semseg_<object>_<run_name>
experiments/<phase_or_task>_<object>_<split_or_run>
```

## 2. Kiểm Tra Môi Trường

Chạy từ project root:

```powershell
python --version
python -m py_compile scripts\generate_synthetic_blender.py scripts\prepare_pointnet2_semseg_dataset.py scripts\validate_raw_dataset.py
python -m py_compile scripts\train_pointnet2_instance_seg.py scripts\eval_pointnet2_instance_seg.py scripts\export_pose_instance_crops.py
python -m py_compile src\data\pose_crop_dataset.py src\models\pointnet2_pose.py src\models\pointnet2_pose_voting.py src\models\pointnet2_pose_refiner.py src\training\pose_losses.py src\training\pose_metrics.py src\training\pose_refiner_losses.py src\inference\pose_refinement.py scripts\train_pointnet2_pose.py scripts\train_pointnet2_pose_refiner.py scripts\eval_pointnet2_pose.py scripts\sweep_pose_crop_export_params.py scripts\run_pose_evaluation_suite.py
```

Kỳ vọng:

```text
Không có output từ py_compile nghĩa là syntax check pass.
```

Nếu dùng CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

## 3. Tạo Raw Synthetic Data

### 3.1. Tạo data bằng GUI

Chạy:

```powershell
python .\scripts\dataset_gui.py
```

Trong group Generation:

- Chọn model STL:
  - `object-model/K41144.stl`
  - hoặc `object-model/bending_pipe.stl`
- Set class name:
  - `K41144`
  - hoặc `bending_pipe`
- Set model scale:
  - K41144: `1.0`
  - bending_pipe: `0.001`
- Chỉnh camera depth, camera RGB, debug camera cho video, đèn, spawn, restitution, settle frame.
- Dùng nút `1 Sample + Video` để tạo đúng một sample kèm `spawn_simulation.mp4`. Chế độ này tự ép video frame step về `1` để dễ quan sát object rơi và settle trước khi generate dataset lớn.
- Dùng Import Preset / Export Preset để lưu cấu hình generator.
- Chọn output folder mới, ví dụ:

```text
synthetic-data/K41144_20260608_large
synthetic-data/bending_pipe_20260608_large
```

### 3.2. Tạo bending_pipe bằng preset CLI

Preset ổn định hiện tại:

```text
configs/generator/bending_pipe_active_spawn_stable.json
```

Chạy Blender background:

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --background --python .\scripts\generate_synthetic_blender.py -- --settings-file .\configs\generator\bending_pipe_active_spawn_stable.json --output .\synthetic-data\bending_pipe_new
```

### 3.3. Tạo K41144 bằng CLI

Ví dụ:

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --background --python .\scripts\generate_synthetic_blender.py -- --model .\object-model\K41144.stl --class-name K41144 --model-scale 1.0 --output .\synthetic-data\K41144_new --samples 80 --objects 30 --width 640 --height 480 --bin-wall-height 0.14 --drop-height-min 0.12 --drop-height-max 0.34 --spawn-strategy layered --objects-per-layer 6 --spawn-min-distance 0.045 --spawn-settle-frames 35 --collision-margin 0.00002 --object-restitution 0.01 --min-visible-objects 12 --min-visible-points 8000 --max-sample-attempts 12 --settle-frames 260
```

Ghi chú generate:

- Khi `--spawn-settle-frames > 0`, generator dùng delayed rigid-body activation. Tất cả object được chuẩn bị ở frame 1, nhưng object chưa tới lượt sẽ bị hidden và disabled rigid body; tới `spawn_frame` thì mới active. Object đã rơi trước đó vẫn active, không bị chuyển sang passive.
- Video debug dùng camera riêng qua `--debug-camera-location`, `--debug-camera-target`, `--debug-camera-lens`; camera này không ảnh hưởng depth/RGB/mask/pose label của dataset.
- Khi tune parameter, nên chạy `--samples 1 --record-simulation-video --simulation-video-frame-step 1` rồi xem `sample_000000/spawn_simulation.mp4`. Video preview dùng material sáng tạm thời và ẩn tường box chỉ trong lúc render video, sau khi RGB/depth/mask/metadata raw đã được ghi xong.

Mỗi sample raw hợp lệ cần có:

```text
depth.png
rgb.png
mask.png
normal.png
point_cloud.ply
metadata.json
```

## 4. Validate Raw Dataset

Chạy sau mỗi lần generate:

```powershell
python .\scripts\validate_raw_dataset.py --data .\synthetic-data\K41144_new --json-output .\experiments\validation_K41144_new.json
python .\scripts\validate_raw_dataset.py --data .\synthetic-data\bending_pipe_new --json-output .\experiments\validation_bending_pipe_new.json
```

Kỳ vọng:

```text
Validation: OK
OK bằng số sample.
```

Reference hiện tại:

```text
synthetic-data/K41144:       77/77 OK
synthetic-data/bending_pipe: 30/30 OK
```

Nếu validation fail:

- Kiểm tra object bị văng khỏi bin.
- Tăng `spawn_min_distance`.
- Tăng `spawn_settle_frames`.
- Tăng `settle_frames`.
- Giảm `object_restitution`.
- Không bật filtering để che lỗi generate nếu dataset dùng cho training chính.

## 5. Build Processed PointNet++ Dataset

Processed dataset được tạo từ raw dataset và có thể regenerate vào folder mới.

K41144:

```powershell
python .\scripts\prepare_pointnet2_semseg_dataset.py --raw .\synthetic-data\K41144 --out .\processed-data\pointnet2_semseg_k41144_new --num-points 16384 --use-normals --object-fraction-target 0.65 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 7
```

bending_pipe:

```powershell
python .\scripts\prepare_pointnet2_semseg_dataset.py --raw .\synthetic-data\bending_pipe --out .\processed-data\pointnet2_semseg_bending_pipe_new --num-points 16384 --use-normals --object-fraction-target 0.65 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 7
```

Folder output:

```text
processed-data/pointnet2_semseg_<object>_new/
  conversion_config.json
  dataset_stats.json
  schema.md
  train/*.npz
  val/*.npz
  test/*.npz
```

Mỗi `.npz` cần có:

```text
points
features
semantic_labels
instance_labels
point_pixels
raw_sample
```

## 5.5. Tune noise/augmentation cho processed data

Noise khi train chỉ chạy trong DataLoader, không được ghi trực tiếp vào `synthetic-data/` hoặc `processed-data/`. Khi cần chỉnh noise cho giống camera thật hơn, dùng app:

```powershell
python .\scripts\noise_tuning_gui.py
```

Workflow khuyến nghị:

- Chọn processed dataset, ví dụ `processed-data/pointnet2_semseg_k41144` hoặc `processed-data/pointnet2_semseg_bending_pipe`.
- Load train config sẽ dùng để train. App tự đọc `dataset.root`, `dataset.normalize`, và block `dataset.augment`.
- Điều chỉnh các tham số jitter, depth noise, point dropout, outlier, normal jitter, z-rotation, quantization, grazing/edge/blob dropout, camera fallback.
- Xem tab VTK `3D Point Cloud` và phần stats: số point bị dịch, mean/p95/max displacement, semantic changed, object->background.
- Dùng `side-by-side` để so clean/noisy tách nhau, hoặc `overlay` để đặt noisy cloud lên clean cloud cùng một góc nhìn. Tab `2D Projection` vẫn giữ lại để kiểm tra nhanh top/front/side.
- Lưu `augment` preset YAML hoặc lưu ra một train config mới. Nút `Update Config` ghi đè block `augment` trong config đang load, nhưng comment YAML có thể bị mất.

App dùng đúng implementation `src/data/augmentation.py` như training. Với `normalize: scene_center`, preview sẽ dùng cloud đã normalize và camera position trong cùng frame, sát với path của instance segmentation DataLoader.

Nếu cần xem 3D bằng Open3D hoặc export PLY clean/noisy:

```powershell
python .\scripts\view_augmented_point_cloud.py .\processed-data\pointnet2_semseg_k41144\test\sample_000011.npz --config .\configs\train\pointnet2_semseg_k41144.yaml
```

### 5.5.1. Real-depth gap từ `real-data/scene_001.pcd` (2026-07-01)

Real reference hiện có là `real-data/scene_001.pcd` của object gần
`bending_pipe`. File này là point cloud PCD binary, 459708 points, tọa độ dạng
mm-like (`z` khoảng `-725..-595`), nên khi đưa vào real-eval phải scale `0.001`
sang mét. Thống kê nhanh sau khi scale: XY nearest-neighbor p50/p90/p99 khoảng
`0.59 / 0.96 / 1.12 mm`; local depth gap giữa 8 hàng xóm XY p50/p90/p95/p99
khoảng `0.85 / 28.27 / 56.89 / 96.55 mm`; khoảng `19.56%` điểm nằm gần depth
discontinuity `> 5 mm`.

Kết luận: synthetic clean-raycast chưa đủ giống real, nhất là biên object sát
nhau và bề mặt dễ mất depth. Với K41144 Wave2, dùng config experiment:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_k41144_wave2_sim2real.yaml --out .\experiments\pointnet2_instance_k41144_wave2_sim2real
```

Luôn so model sim-to-real với clean checkpoint trên cùng synthetic gate và real
set đã label. Không ghi noise trực tiếp vào `synthetic-data/`; raw clean vẫn là
nguồn GT/mask/pose để audit.

Generator và GUI hiện cũng có raw sensor-noise mode để tạo một wave riêng gần
real hơn ngay từ bước export raw. Default vẫn là clean:

```powershell
python .\scripts\generate_synthetic_blender.py --settings-file .\configs\generator\k41144_grid.json --sensor-noise-profile structured_light --output .\synthetic-data\k41144_sensor_noise_smoke --samples 3
```

Các parameter chính: `sensor_noise_profile`, `depth_quantization_m`,
`depth_quadratic_noise`, `depth_random_dropout_prob`, `edge_gap_m`,
`edge_dropout_prob`, `edge_smear_prob`, `edge_smear_radius_px`,
`flying_pixel_prob`, `flying_pixel_alpha_min/max`, `bridge_prob`,
`bridge_max_gap_px`, `blob_dropout_count`, `blob_dropout_radius_px`. GUI có group
`Depth Sensor Noise` để chỉnh/export/import các giá trị này. Khi profile khác
`none`, `sensor_data.npz` lưu thêm `sensor_artifact_mask` và `metadata.json` lưu
`sensor_noise_summary`.

## 6. Train Instance Segmentation

### 6.1. Smoke test trước khi train full

K41144:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144 --epochs 2 --batch-size 1 --device cuda --limit-train-samples 2 --limit-val-samples 2 --disable-augment
```

bending_pipe:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_bending_pipe.yaml --data .\processed-data\pointnet2_semseg_bending_pipe --epochs 2 --batch-size 1 --device cuda --limit-train-samples 2 --limit-val-samples 2 --disable-augment
```

Kỳ vọng:

```text
[train] epoch=1 ...
[train] epoch=2 ...
experiments/pointnet2_instance_<object>_<timestamp>/checkpoints/best.pt tồn tại.
```

### 6.2. Train full instance model

K41144:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_k41144.yaml --data .\processed-data\pointnet2_semseg_k41144 --device cuda
```

bending_pipe:

```powershell
python .\scripts\train_pointnet2_instance_seg.py --config .\configs\train\pointnet2_instance_bending_pipe.yaml --data .\processed-data\pointnet2_semseg_bending_pipe --device cuda
```

Reference checkpoint hiện tại:

```text
K41144:
  experiments/pointnet2_instance_k41144_20260607_165729/checkpoints/best.pt

bending_pipe:
  experiments/pointnet2_instance_bending_pipe_20260607_171218/checkpoints/best.pt
```

## 7. Evaluate Instance Segmentation

K41144:

```powershell
python .\scripts\eval_pointnet2_instance_seg.py --checkpoint .\experiments\pointnet2_instance_k41144_20260607_165729\checkpoints\best.pt --data .\processed-data\pointnet2_semseg_k41144 --split test --out .\experiments\pointnet2_instance_k41144_20260607_165729\test_metrics.json --device cuda --ops-backend pytorch3d --dbscan-eps-m 0.004 --min-cluster-points 96 --preview-samples 3
```

bending_pipe:

```powershell
python .\scripts\eval_pointnet2_instance_seg.py --checkpoint .\experiments\pointnet2_instance_bending_pipe_20260607_171218\checkpoints\best.pt --data .\processed-data\pointnet2_semseg_bending_pipe --split test --out .\experiments\pointnet2_instance_bending_pipe_20260607_171218\test_metrics.json --device cuda --ops-backend pytorch3d --dbscan-eps-m 0.006 --min-cluster-points 32 --preview-samples 3
```

Metric cần xem:

```text
overall_semantic_metrics.object_iou
mean_instance_metrics.instance_recall
mean_instance_metrics.instance_precision
mean_instance_metrics.instance_mean_iou
mean_instance_metrics.merge_count
mean_instance_metrics.split_count
```

Các file preview PLY giúp debug lỗi merge/split instance.

## 8. Export Pose Crops Từ GT Instances

Pose training không đọc raw Blender trực tiếp. Cần export crop theo instance trước.

K41144:

```powershell
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_k41144 --split train --source gt --raw-root .\synthetic-data\K41144 --out .\experiments\pose_crops_k41144_gt_train_xyznormal
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_k41144 --split val --source gt --raw-root .\synthetic-data\K41144 --out .\experiments\pose_crops_k41144_gt_val_xyznormal
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_k41144 --split test --source gt --raw-root .\synthetic-data\K41144 --out .\experiments\pose_crops_k41144_gt_test_xyznormal
```

bending_pipe:

```powershell
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_bending_pipe --split train --source gt --raw-root .\synthetic-data\bending_pipe --out .\experiments\pose_crops_bending_pipe_gt_train_xyznormal
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_bending_pipe --split val --source gt --raw-root .\synthetic-data\bending_pipe --out .\experiments\pose_crops_bending_pipe_gt_val_xyznormal
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_bending_pipe --split test --source gt --raw-root .\synthetic-data\bending_pipe --out .\experiments\pose_crops_bending_pipe_gt_test_xyznormal
```

Reference GT crops:

```text
K41144:
  train: experiments/phase14_pose_crops_k41144_gt_train_xyznormal   1406 crops
  val:   experiments/phase14_pose_crops_k41144_gt_val_xyznormal     185 crops
  test:  experiments/phase14_pose_crops_k41144_gt_test_xyznormal    146 crops

bending_pipe:
  train: experiments/phase14_pose_crops_bending_pipe_gt_train_xyznormal 142 crops
  val:   experiments/phase14_pose_crops_bending_pipe_gt_val_xyznormal   18 crops
  test:  experiments/phase14_pose_crops_bending_pipe_gt_test_xyznormal  18 crops
```

## 9. Kiểm Tra Pose Dataset Bằng Identity Baseline

Chạy sau mọi thay đổi frame, scale, hoặc export pose crop.

K41144:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --data .\experiments\phase14_pose_crops_k41144_gt_val_xyznormal --out .\experiments\pose_identity_k41144_val.json --identity-baseline --device cuda
```

bending_pipe:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --data .\experiments\phase14_pose_crops_bending_pipe_gt_val_xyznormal --out .\experiments\pose_identity_bending_pipe_val.json --identity-baseline --device cuda
```

Kỳ vọng:

```text
ADD gần 0 mm
translation gần 0 mm
pose_success_add_0.1d = 1.0
```

## 10. Train Pose Keypoint/Center Voting

Checkpoint quan trọng nhất là:

```text
checkpoints/best_add.pt
```

Sử dụng `best_add.pt` để evaluate pose vì checkpoint này được chọn theo validation ADD.

K41144:

```powershell
python .\scripts\train_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --data .\experiments\phase14_pose_crops_k41144_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_k41144_gt_val_xyznormal --epochs 30 --batch-size 32 --device cuda --learning-rate 0.001 --dropout 0.05
```

bending_pipe:

```powershell
python .\scripts\train_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --data .\experiments\phase14_pose_crops_bending_pipe_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_bending_pipe_gt_val_xyznormal --epochs 90 --batch-size 16 --device cuda --learning-rate 0.001 --dropout 0.05
```

Fine-tune K41144 từ checkpoint cũ:

```powershell
python .\scripts\train_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --data .\experiments\phase14_pose_crops_k41144_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_k41144_gt_val_xyznormal --epochs 40 --batch-size 32 --device cuda --learning-rate 0.0005 --dropout 0.05 --resume .\experiments\pointnet2_pose_k41144_20260608_021636\checkpoints\best_add.pt --resume-new-experiment
```

Fine-tune bending_pipe:

```powershell
python .\scripts\train_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --data .\experiments\phase14_pose_crops_bending_pipe_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_bending_pipe_gt_val_xyznormal --epochs 150 --batch-size 16 --device cuda --learning-rate 0.0002 --dropout 0.05 --resume .\experiments\pointnet2_pose_bending_pipe_20260608_015529\checkpoints\best_add.pt --resume-new-experiment
```

Reference pose checkpoint hiện tại:

```text
K41144:
  experiments/pointnet2_pose_k41144_20260608_023605/checkpoints/best_add.pt

bending_pipe:
  experiments/pointnet2_pose_bending_pipe_20260608_021309/checkpoints/best_add.pt
```

## 11. Evaluate Pose Trên GT Crops

K41144:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\pointnet2_pose_k41144_20260608_023605\test_best_add_eval.json --device cuda
```

bending_pipe:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_bending_pipe_gt_test_xyznormal --out .\experiments\pointnet2_pose_bending_pipe_20260608_021309\test_best_add_eval.json --device cuda
```

Reference raw GT-crop:

```text
K41144:
  ADD 5.607 mm
  translation 6.021 mm
  ADD_0.1d success 0.897

bending_pipe:
  ADD 9.899 mm
  translation 9.529 mm
  ADD_0.1d success 0.833
```

## 12. Export Predicted Pose Crops Để Chạy Full Pipeline

Đây là bước chạy instance model trước, lấy predicted instance, rồi export thành crop cho pose model.

K41144:

```powershell
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_k41144 --split test --source predicted --checkpoint .\experiments\pointnet2_instance_k41144_20260607_165729\checkpoints\best.pt --raw-root .\synthetic-data\K41144 --out .\experiments\phase17_pose_crops_k41144_pred_test_xyznormal --device cuda --ops-backend pytorch3d --dbscan-eps-m 0.004 --min-cluster-points 96
```

bending_pipe:

```powershell
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_bending_pipe --split test --source predicted --checkpoint .\experiments\pointnet2_instance_bending_pipe_20260607_171218\checkpoints\best.pt --raw-root .\synthetic-data\bending_pipe --out .\experiments\phase17_pose_crops_bending_pipe_pred_test_xyznormal --device cuda --ops-backend pytorch3d --dbscan-eps-m 0.006 --min-cluster-points 32
```

Reference predicted crop:

```text
K41144:
  total_crops 150
  crops_with_object_to_camera 150

bending_pipe:
  total_crops 14
  crops_with_object_to_camera 14
```

## 13. Evaluate Pose Trên Predicted Crops

Luôn chạy cả filtered và all predicted.

K41144 filtered:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase17_pose_crops_k41144_pred_test_xyznormal --out .\experiments\pointnet2_pose_k41144_20260608_023605\pred_test_iou05_eval.json --min-matched-gt-iou 0.5 --device cuda
```

K41144 all:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase17_pose_crops_k41144_pred_test_xyznormal --out .\experiments\pointnet2_pose_k41144_20260608_023605\pred_test_all_eval.json --device cuda
```

bending_pipe filtered:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase17_pose_crops_bending_pipe_pred_test_xyznormal --out .\experiments\pointnet2_pose_bending_pipe_20260608_021309\pred_test_iou05_eval.json --min-matched-gt-iou 0.5 --device cuda
```

bending_pipe all:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase17_pose_crops_bending_pipe_pred_test_xyznormal --out .\experiments\pointnet2_pose_bending_pipe_20260608_021309\pred_test_all_eval.json --device cuda
```

Reference predicted-crop raw:

```text
K41144 IoU>=0.5:
  ADD 10.719 mm
  translation 11.087 mm
  ADD_0.1d 0.738

K41144 all:
  ADD 21.932 mm
  translation 22.831 mm
  ADD_0.1d 0.540

bending_pipe IoU>=0.5:
  ADD 12.146 mm
  translation 13.470 mm
  ADD_0.1d 0.923

bending_pipe all:
  ADD 18.456 mm
  translation 18.612 mm
  ADD_0.1d 0.714
```

## 14. Optional Pose Refinement Bằng Hybrid ICP

Refinement không thay thế raw model. Nó chỉ tạo refined metrics để báo cáo riêng.

Preset:

```text
configs/eval/pose_refinement_defaults.json
```

K41144 GT refined:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\phase20_refine_hybrid_wide16_t20_r20_k41144_gt.json --device cuda --refine-pose --refinement-method hybrid --refinement-distance-threshold-fraction 0.16 --refinement-max-translation-delta-fraction 0.20 --refinement-max-rotation-delta-deg 20
```

bending_pipe GT refined:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_bending_pipe_gt_test_xyznormal --out .\experiments\phase20_refine_hybrid_wide16_t20_r20_bending_gt.json --device cuda --refine-pose --refinement-method hybrid --refinement-distance-threshold-fraction 0.16 --refinement-max-translation-delta-fraction 0.20 --refinement-max-rotation-delta-deg 20
```

Reference GT-crop refined:

```text
K41144:
  raw     ADD 5.607 mm, translation 6.021 mm, ADD_0.1d 0.897
  refined ADD 4.896 mm, translation 5.311 mm, ADD_0.1d 0.925

bending_pipe:
  raw     ADD 9.899 mm, translation 9.529 mm, ADD_0.1d 0.833
  refined ADD 5.870 mm, translation 5.282 mm, ADD_0.1d 0.944
```

Lưu ý:

```text
K41144 predicted-all translation tăng nhẹ sau refinement dù ADD và ADD_0.1d tốt hơn.
Không dùng refinement để che lỗi crop quality.
```

## 15. Sweep Predicted-Crop Parameters Cho K41144

Mục tiêu: giảm crop xấu trong full pipeline K41144.

Full sweep:

```powershell
python .\scripts\sweep_pose_crop_export_params.py --instance-checkpoint .\experiments\pointnet2_instance_k41144_20260607_165729\checkpoints\best.pt --instance-data .\processed-data\pointnet2_semseg_k41144 --raw-root .\synthetic-data\K41144 --pose-config .\configs\train\pointnet2_pose_voting_k41144.yaml --pose-checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --out .\experiments\phase21_k41144_pred_crop_sweep.json --device cuda --refine-pose
```

Smoke:

```powershell
python .\scripts\sweep_pose_crop_export_params.py --instance-checkpoint .\experiments\pointnet2_instance_k41144_20260607_165729\checkpoints\best.pt --instance-data .\processed-data\pointnet2_semseg_k41144 --raw-root .\synthetic-data\K41144 --pose-config .\configs\train\pointnet2_pose_voting_k41144.yaml --pose-checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --out .\experiments\phase21_smoke_k41144_pred_crop_sweep.json --object-probability-thresholds 0.50 --dbscan-eps-m 0.004 --min-cluster-points 96 --limit-samples 1 --device cuda --refine-pose
```

Summary JSON sẽ có:

```text
recommended
rows[].params
rows[].crop_summary
rows[].pose_all
rows[].pose_matched
```

## 16. Train Learned Translation Refiner

Refiner học residual translation từ raw pose checkpoint đã freeze. Không thay backbone PointNet++ chính.

K41144:

```powershell
python .\scripts\train_pointnet2_pose_refiner.py --config .\configs\train\pointnet2_pose_refiner_k41144.yaml --data .\experiments\phase14_pose_crops_k41144_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_k41144_gt_val_xyznormal --device cuda
```

bending_pipe:

```powershell
python .\scripts\train_pointnet2_pose_refiner.py --config .\configs\train\pointnet2_pose_refiner_bending_pipe.yaml --data .\experiments\phase14_pose_crops_bending_pipe_gt_train_xyznormal --val-data .\experiments\phase14_pose_crops_bending_pipe_gt_val_xyznormal --device cuda
```

Evaluate với learned refiner:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\k41144_gt_with_translation_refiner.json --translation-refiner-checkpoint .\experiments\<refiner_run>\checkpoints\best_add.pt --device cuda
```

Evaluate với learned refiner rồi thêm hybrid ICP:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\phase14_pose_crops_k41144_gt_test_xyznormal --out .\experiments\k41144_gt_with_translation_refiner_and_icp.json --translation-refiner-checkpoint .\experiments\<refiner_run>\checkpoints\best_add.pt --device cuda --refine-pose --refinement-method hybrid --refinement-distance-threshold-fraction 0.16 --refinement-max-translation-delta-fraction 0.20 --refinement-max-rotation-delta-deg 20
```

Hiện tại learned refiner vẫn là experimental. Chỉ promote nếu full evaluation suite chứng minh tốt hơn Phase 20 hybrid refinement trên cả hai object.

## 17. Chạy Full Pose Evaluation Suite

Đây là lệnh nên dùng sau mọi thay đổi lớn về data, checkpoint, crop quality, pose model hoặc refinement.

Full suite raw/refined:

```powershell
python .\scripts\run_pose_evaluation_suite.py --device cuda --refine-pose
```

Smoke suite:

```powershell
python .\scripts\run_pose_evaluation_suite.py --out-dir .\experiments\phase23_smoke_pose_eval_suite --device cuda --limit-samples 2 --refine-pose
```

Output:

```text
experiments/pose_eval_suite_<timestamp>/
  summary.json
  summary.md
  K41144_identity_val.json
  K41144_gt_test.json
  K41144_pred_test_iou05.json
  K41144_pred_test_all.json
  bending_pipe_identity_val.json
  bending_pipe_gt_test.json
  bending_pipe_pred_test_iou05.json
  bending_pipe_pred_test_all.json
```

Nếu muốn suite kèm learned refiner:

```powershell
python .\scripts\run_pose_evaluation_suite.py --device cuda --refine-pose --k41144-translation-refiner-checkpoint .\experiments\<k41144_refiner>\checkpoints\best_add.pt --bending-translation-refiner-checkpoint .\experiments\<bending_refiner>\checkpoints\best_add.pt
```

## 18. Cách Chạy Model End-To-End

Với synthetic/test data hiện tại, cách chạy full model là:

```text
processed point cloud
  -> instance model predict semantic/instance
  -> export predicted pose crops
  -> pose model predict object_to_camera
  -> optional refinement
  -> JSON metrics và per-crop samples
```

### 18.1. K41144 end-to-end

1. Export predicted crops:

```powershell
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_k41144 --split test --source predicted --checkpoint .\experiments\pointnet2_instance_k41144_20260607_165729\checkpoints\best.pt --raw-root .\synthetic-data\K41144 --out .\experiments\run_k41144_pred_crops --device cuda --ops-backend pytorch3d --dbscan-eps-m 0.004 --min-cluster-points 96
```

2. Run pose raw:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\run_k41144_pred_crops --out .\experiments\run_k41144_pose_raw.json --device cuda
```

3. Run pose refined:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_k41144.yaml --checkpoint .\experiments\pointnet2_pose_k41144_20260608_023605\checkpoints\best_add.pt --data .\experiments\run_k41144_pred_crops --out .\experiments\run_k41144_pose_refined.json --device cuda --refine-pose --refinement-method hybrid --refinement-distance-threshold-fraction 0.16 --refinement-max-translation-delta-fraction 0.20 --refinement-max-rotation-delta-deg 20
```

### 18.2. bending_pipe end-to-end

1. Export predicted crops:

```powershell
python .\scripts\export_pose_instance_crops.py --data .\processed-data\pointnet2_semseg_bending_pipe --split test --source predicted --checkpoint .\experiments\pointnet2_instance_bending_pipe_20260607_171218\checkpoints\best.pt --raw-root .\synthetic-data\bending_pipe --out .\experiments\run_bending_pipe_pred_crops --device cuda --ops-backend pytorch3d --dbscan-eps-m 0.006 --min-cluster-points 32
```

2. Run pose raw:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\run_bending_pipe_pred_crops --out .\experiments\run_bending_pipe_pose_raw.json --device cuda
```

3. Run pose refined:

```powershell
python .\scripts\eval_pointnet2_pose.py --config .\configs\train\pointnet2_pose_voting_bending_pipe.yaml --checkpoint .\experiments\pointnet2_pose_bending_pipe_20260608_021309\checkpoints\best_add.pt --data .\experiments\run_bending_pipe_pred_crops --out .\experiments\run_bending_pipe_pose_refined.json --device cuda --refine-pose --refinement-method hybrid --refinement-distance-threshold-fraction 0.16 --refinement-max-translation-delta-fraction 0.20 --refinement-max-rotation-delta-deg 20
```

Trong output JSON, xem:

```text
raw_metrics
refined_metrics
refinement.summary
samples[].raw_metrics
samples[].refined_metrics
```

## 19. Checklist Khi Tạo Dataset Lớn Mới

1. Generate raw data mới vào folder mới trong `synthetic-data/`.
2. Validate raw dataset.
3. Convert sang processed PointNet++ dataset.
4. Train instance model.
5. Evaluate instance model trên test split.
6. Export GT pose crops train/val/test.
7. Identity baseline cho pose crops.
8. Train pose keypoint/center voting.
9. Evaluate GT-crop pose.
10. Export predicted pose crops.
11. Evaluate predicted-crop pose filtered và all.
12. Run optional hybrid refinement.
13. Nếu K41144 predicted-all còn yếu, chạy Phase 21 sweep.
14. Nếu translation vẫn không đạt target, train learned translation refiner.
15. Chạy Phase 23 full suite không `--limit-samples`.

## 20. Troubleshooting Nhanh

### Raw data validate fail

- Tăng `spawn_settle_frames`.
- Tăng `settle_frames`.
- Giảm `object_restitution`.
- Tăng `spawn_min_distance`.
- Kiểm tra sample có object văng khỏi bin.

### Instance segmentation nhiều merge

- Giảm `dbscan_eps_m`.
- Tăng `min_cluster_points` nếu có nhiều fragment nhỏ sai.
- Kiểm tra `instance_mean_iou`, `merge_count`, `split_count`.
- Xuất preview PLY để xem cluster.

### Predicted-crop pose yếu nhưng GT-crop pose tốt

- Vấn đề chính là instance crop quality.
- Chạy Phase 21 sweep cho K41144.
- Báo cáo riêng `matched_gt_iou >= 0.5` và all predicted.

### Identity pose không gần 0

- Kiểm tra `model_scale`.
- Kiểm tra frame camera và flip `z`.
- Kiểm tra `raw-root` đúng với processed dataset.
- Không train tiếp khi identity baseline fail.

### Refined pose tốt hơn ADD nhưng translation xấu hơn

- Không dùng refined pose thay raw pose.
- Report raw/refined riêng.
- Kiểm tra `refinement.summary.catastrophic_add_worse_0.05d_count`.
- Với K41144 predicted-all, đây là rủi ro đã biết.

## 21. Trạng Thái Reference Hiện Tại

> Cho bending_pipe dùng checkpoint Wave 2 dưới đây (xem section "Cập Nhật 2026-06-16"). Checkpoint 2026-06-07/08 chỉ giữ làm lịch sử. K41144 vẫn dùng 2026-06-08 cho tới khi có data lớn.

Instance checkpoints:

```text
K41144:
  experiments/pointnet2_instance_k41144_20260607_165729/checkpoints/best.pt
bending_pipe (Wave 2, dùng cái này):
  experiments/pointnet2_instance_bending_pipe_wave2_20260614_154448/checkpoints/best.pt
bending_pipe (cũ, lịch sử):
  experiments/pointnet2_instance_bending_pipe_20260607_171218/checkpoints/best.pt
```

Pose checkpoints:

```text
K41144:
  experiments/pointnet2_pose_k41144_20260608_023605/checkpoints/best_add.pt
bending_pipe (Wave 2, dùng cái này — GT-crop ADD 4.57 mm / ADD_0.1d 0.960):
  experiments/pointnet2_pose_bending_pipe_20260614_191629/checkpoints/best_add.pt
bending_pipe (cũ, lịch sử):
  experiments/pointnet2_pose_bending_pipe_20260608_021309/checkpoints/best_add.pt
```

Bundle đã đóng gói (registry, gitignored — dựng lại bằng package_model_bundle.py):

```text
models/bending_pipe/v2  (latest; instance Wave 2 + pose Wave 2 + model_fit confidence)
models/K41144/v1
```

Phase 20 refined GT-crop reference:

```text
K41144:
  raw     ADD 5.607 mm, translation 6.021 mm, ADD_0.1d 0.897
  refined ADD 4.896 mm, translation 5.311 mm, ADD_0.1d 0.925

bending_pipe:
  raw     ADD 9.899 mm, translation 9.529 mm, ADD_0.1d 0.833
  refined ADD 5.870 mm, translation 5.282 mm, ADD_0.1d 0.944
```

Phase 23 smoke output:

```text
experiments/phase23_smoke_pose_eval_suite/summary.json
experiments/phase23_smoke_pose_eval_suite/summary.md
```

Việc tiếp theo:

```text
1. bending_pipe: ĐÃ XONG Wave 2 (instance + pose retrain trên data lớn) + đóng gói bundle v2 + backend.
2. K41144: lặp lại Wave 2 khi có data lớn — gộp multi-root, retrain instance + pose,
   đóng gói bundle K41144 v2 (giống quy trình bending_pipe ở section "Cập Nhật 2026-06-16").
3. KHÔNG train pose from-scratch trên predicted crops (đã kiểm chứng tệ hơn — xem mục D).
   Nếu cần nâng per-instance precision: finetune từ GT model hoặc mix GT+predicted.
4. Chạy full Phase 23 suite không dùng --limit-samples sau mỗi lần retrain.
5. Chỉ promote checkpoint/bundle nếu eval (GT-crop gate + dense-pile + suite) cải thiện ổn định.
6. Khi có camera thật: chạy P8 real-eval harness + cân nhắc bật sim-to-real noise augmentation.
```
