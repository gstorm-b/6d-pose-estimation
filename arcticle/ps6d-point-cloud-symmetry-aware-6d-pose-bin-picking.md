# PS6D: Point Cloud Based Symmetry-Aware 6D Object Pose Estimation In Robot Bin-Picking

Nguon: Yifan Yang, Zhihao Cui, Qianyi Zhang, Jingtai Liu, "PS6D: Point Cloud Based Symmetry-Aware 6D Object Pose Estimation in Robot Bin-Picking", arXiv:2405.11257v1, 2024.

Link: https://arxiv.org/html/2405.11257v1

## 1. Muc tieu bai bao

Bai bao de xuat PS6D, mot framework uoc luong 6D pose dua tren point cloud cho robot bin-picking. Bai toan muc tieu la cac chi tiet cong nghiep it texture, co be mat phan quang, bi xep chong, bi che khuat, co hinh dang dai/manh va co doi xung.

Khac voi cac method dua vao RGB hoac RGB-D fusion, PS6D chi dung point cloud. Ly do phu hop voi bin-picking cong nghiep:

- Vat cong nghiep thuong it texture, mau sac khong on dinh giua cac batch.
- Be mat phan quang, vat mau den, hoac vat co do tuong phan thap lam RGB khong dang tin.
- Point cloud giu truc tiep thong tin hinh hoc, phu hop hon cho pose cua chi tiet co CAD.

Dau ra cuoi cung cua PS6D la instance segmentation va 6D pose cho tung object trong scene.

## 2. Y tuong cot loi

PS6D xem bin-picking nhu bai toan point-wise voting:

1. Normalize scene point cloud vao mot khong gian lam viec chuan.
2. Dung PointNet++ lam backbone, bo sung attention/Point Transformer de hoc quan he local-global.
3. Voi moi point, predict:
   - offset den centroid cua instance chua point do.
   - quaternion/rotation cua instance.
4. Dua translation vote va rotation vote vao two-stage clustering.
5. Vote pose cuoi cung cho tung instance.
6. Trong robot pipeline, pose sau network duoc refine bang ICP va chuyen thanh grasp pose.

Diem quan trong: instance segmentation khong tach roi pose estimation. Instance duoc suy ra tu cac vote pose/centroid cua tung point, nen clustering co them tin hieu hinh hoc va rotation, khong chi dua vao raw xyz.

## 3. Van de bai bao muon giai quyet

### 3.1 Slender objects

Voi object dai/manh, hai object co the cat nhau gan centroid. Neu chi cluster theo centroid vote, diem cua hai object co the bi gom nham thanh mot cluster. Trung binh pose cua cluster bi merge thuong khong phu hop voi bat ky object nao.

### 3.2 Multi-symmetric objects

Vat doi xung co nhieu rotation tuong duong ve mat quan sat. Neu loss coi chi mot rotation label la dung, network bi phat oan khi predict mot pose tuong duong. Neu lay trung binh cac rotation khong tuong thich, pose ket qua co the sai vat ly.

### 3.3 Industrial domain gap

Public dataset nhu LineMOD/YCB khong dai dien tot cho bin-picking cong nghiep. PS6D vi vay danh gia tren Sileane, IPA va mot tap PS6D dataset gom cac workpiece cong nghiep.

## 4. Kien truc PS6D

Input:

```text
scene point cloud
```

Paper noi network chi can point position. Voi project cua minh, co the van giu `xyz + normal_camera` vi current semantic baseline da dung normals va dat metric tot.

Backbone:

- PointNet++ Set Abstraction de downsample va gom local neighborhood.
- Point Transformer/attention de hoc trong so quan he giua cac point.
- Feature Propagation de upsample feature ve tung point.

Output point-wise:

```text
centroid offset / centroid position vote
rotation quaternion
```

Loss tong hop:

```text
total_loss = translation_loss + rotation_loss
```

Trong paper, segmentation la ket qua cua clustering tren cac point-wise pose votes. Voi project hien tai, co the implement theo tung buoc:

1. Instance segmentation branch truoc:
   - semantic logits
   - centroid offsets
   - instance embeddings
   - DBSCAN tren voted centers
2. Pose branch sau:
   - rotation quaternion
   - symmetry-aware rotation loss
   - two-stage clustering dung ca translation va rotation

## 5. Normalized Workpiece Space

PS6D normalize object model point cloud vao cube kich thuoc 100 mm va dua scene point cloud ve gan origin. Muc tieu:

- Giam phu thuoc vao kich thuoc vat the.
- Giam phu thuoc vao camera pose/range.
- Cho phep mot bo tham so SA radius/backbone ap dung cho nhieu object.

Lien he voi project:

- K41144 co kich thuoc xap xi `30 mm x 93.6 mm x 20 mm`, cung la dang vat dai/manh.
- Current PointNet++ pipeline dang dung `normalize: scene_center`.
- Khi sang pose estimation, can quyet dinh ro:
  - train offset/pose trong camera metric space, hay
  - train trong normalized workpiece/scene space roi map nguoc ve camera frame.

Khuyen nghi:

- Phase instance segmentation: tiep tuc dung `scene_center` de giu don gian va tuong thich semantic baseline.
- Phase pose regression: them metadata normalization transform ro rang, vi pose can map nguoc ve `object_to_camera`.
- Ten bien nen ro frame, vi project da co rule bat buoc ve frame direction.

## 6. Center Distance Sensitive Translation Loss

PS6D predict centroid tu moi point. Voi object dai, point o gan dau object cach centroid xa hon va kho regress hon. Paper them weighting theo khoang cach point-centroid, normalize trong khoang xap xi `[0.5, 1.5]`.

Y nghia:

- Point xa centroid duoc loss weight cao hon.
- Network hoc tot hon voi slender objects.
- Giam loi merge khi cac object giao nhau gan trung tam.

Lien he voi plan instance segmentation hien tai:

```text
target_offset = instance_centroid_camera - point_camera
```

Nen them tuy chon loss weight:

```text
offset_weight_per_point = normalized_distance(point, instance_centroid)
```

De MVP khong qua phuc tap:

1. Bat dau voi Smooth L1 offset loss khong weight.
2. Neu object bi merge/split o vung dau vat hoac offset hoi tu kem, them center-distance-sensitive weight.
3. Track `offset_l1_object` rieng cho cac point xa centroid.

## 7. Symmetry-Aware Rotation Loss

PS6D xu ly symmetry bang cach ghi lai thong tin doi xung cua object truoc khi train:

```text
symmetry degree quanh x/y/z
finite symmetry matrices
infinite symmetry axis neu co
```

Network predict quaternion, nhung loss chuyen quaternion sang rotation matrix de so sanh vi quaternion co tinh khong lien thong dau `q` va `-q`.

Voi finite symmetry:

- Tao danh sach rotation symmetry hop le.
- Chon symmetry transform cho loss nho nhat.

Voi infinite symmetry:

- Khong phat rotation quanh truc doi xung.
- Chi so sanh phan hinh hoc co y nghia quan sat.

Lien he voi K41144:

- Can audit CAD/STL de xac dinh object co doi xung that su khong.
- Neu co doi xung gan-dung nhung khong hoan hao, phai quyet dinh theo grasp requirement:
  - neu robot grasp khong phan biet hai pose tuong duong, coi la symmetry.
  - neu grasp/assembly can orientation tuyet doi, khong nen bo qua truc do trong loss.

Day la phan nen de danh dau la pose-stage decision, chua nen chen vao instance MVP.

## 8. Two-Stage Clustering

PS6D de xuat two-stage clustering de giai quyet ca slender object va symmetry.

Stage 1:

- Cluster dua tren predicted centroid position va predicted rotation.
- Muc tieu la tach cac object giao nhau, nhat la khi centroid vote gan nhau nhung rotation khac nhau.

Stage 2:

- Cluster lai dua tren centroid position cua cac cluster stage 1.
- Muc tieu la merge cac cluster cung instance bi tach ra do symmetry/rotation ambiguity.
- Sau do pose voting chon quaternion tot nhat trong nhom.

Lien he voi project:

Current instance plan de xuat:

```text
voted_center = point_xyz + predicted_offset
DBSCAN(voted_center)
```

Do la stage 1 don gian va hop ly cho MVP instance segmentation. Sau khi them rotation head, co the nang cap:

```text
stage 1: cluster(voted_center + rotation_embedding/quaternion)
stage 2: merge cluster means by center distance
pose vote: choose/average pose with symmetry-aware rule
```

Can can than voi quaternion distance:

- `q` va `-q` la cung rotation.
- Voi symmetric object, nhieu rotation khac nhau van tuong duong.
- Khong nen dua raw quaternion vao DBSCAN truoc khi co distance metric on.

## 9. Dataset Generation

PS6D dung Blender/CAD simulation:

- Tha object tu do cao ngau nhien vao bin.
- Giu physics collision de tao scattered/stacked scenes.
- Render/thu point cloud sau khi vat on dinh.
- Tao ground truth pose tu simulation.

Paper cung noi visibility information ton thoi gian, va ho khong co visibility module trong network, nen dung sparse stacked data hoac sparsification augmentation khi train.

Lien he voi repo:

- Current generator da dung Blender physics va da co strict out-of-bin rejection.
- Raw data da luu `depth_m`, `instance_mask`, `normal_camera`, `points_camera`, `object_to_camera`.
- Processed semantic dataset da reconstruct full-scene points tu depth, dung `instance_mask` de label.

Khoang cach can bu:

- Instance/pose training can luu pose target gan voi `instance_id`.
- Neu cluster/pose eval bo qua object bi che khuat nang, phai co rule `min_points_per_instance` hoac visibility proxy.
- PS6D dung visibility threshold 0.4 trong metric; project minh hien co the dung point-count threshold truoc, sau do tinh visibility bang visible model points / full model sampled points neu can.

## 10. Evaluation Metrics

PS6D dung hai nhom metric:

### 10.1 Instance-level F-score

Metric nay xem predicted pose/instance co match voi ground truth instance khong. Noi cach khac, no phat hien:

- missed detections
- false detections
- cluster count sai

Paper chi tinh cac instance du visible du lon, vi vat bi che khuat nang thuong khong phai grasp candidate tot.

### 10.2 Point-wise Recall

Metric nay do chat luong pose tren tung point cua object, co tinh den symmetry. No phan anh predicted pose co align dung visible object geometry khong.

Lien he voi MVP instance segmentation:

- Truoc pose stage, dung Hungarian matching voi point-set IoU la hop ly.
- Track `instance_precision`, `instance_recall`, `instance_mean_iou`.
- Report rieng ignored/tiny GT instances, khong am tham bo qua.

Khi sang pose stage:

- Them pose distance metric theo model points.
- Them symmetry-aware ADD/ADD-S hoac metric tuong tu PS6D.
- Tach eval thanh:
  - pose with GT instances
  - pose with predicted instances
  - full pipeline pose

## 11. Ket qua chinh

PS6D duoc so sanh voi PPF, PPR-Net va PPR-Net++ tren Sileane, IPA va PS6D dataset.

Ket qua trung binh trong bang cua paper:

```text
PPF:       F = 51.59, Recall = 41.50
PPR-Net:   F = 84.76, Recall = 78.52
PPR-Net++: F = 87.53, Recall = 80.11
PS6D:      F = 97.60, Recall = 91.93
```

Tac gia report PS6D hon PPR-Net++:

```text
F:      +11.5%
Recall: +14.8%
```

Vat dang chu y:

- `PS6D_23` la object rat dai, bbox `[32, 92, 1304] mm`; cac baseline deep learning kem hon nhieu, PS6D van dat F/Recall cao hon ro.
- `PS6D_24` la object co multiple symmetries; PS6D dat gan hoan hao trong bang.

Robot bin-picking real-world:

- Train tren simulated CAD dataset.
- Inference tren point cloud real scene.
- Dung Mech-Eye ProS camera, Fanuc robot, ICP refinement.
- Report grasping success rate `91.7%`.

## 12. Ablation Study

Paper thay the/bo cac thanh phan:

- Doi backbone ve PointNet2MSG.
- Bo normalization.
- Dung single-stage clustering.

Ket luan:

- Attention/Transformer tren PointNet++ giup tang performance, nhat la case kho.
- Normalization quan trong khi train nhieu object kich thuoc khac nhau.
- Two-stage clustering dac biet quan trong voi slender va multi-symmetric objects.

Lien he voi K41144:

- Vi K41144 dai va co kha nang cham/xep chong, clustering la rui ro lon nhat cua phase instance segmentation.
- Mot semantic IoU rat cao khong dam bao instance separation tot.
- Can xem PLY preview sau moi run; chi metric semantic khong du.

## 13. Diem nen hoc de implement instance segmentation

### 13.1 Nen giu PointNet++ backbone hien tai

Backbone semantic da hoat dong tot trong repo. PS6D cung dung PointNet++ lam nen, nen huong dung lai backbone la dung.

MVP nen them head:

```text
semantic_logits: [B, N, 2]
offsets:         [B, N, 3]
embeddings:      [B, N, D]
```

Sau do train:

```text
semantic CE
object-only Smooth L1 centroid offset
discriminative embedding loss
```

### 13.2 Offset voting nen la tin hieu chinh

Embedding giup tach instance, nhung voi vat cung class cham nhau, voted center se la tin hieu rat quan trong. Do do plan hien tai "offset la primary, embedding weight thap" la hop ly.

### 13.3 Them center-distance-sensitive weighting sau MVP

Neu overfit/debug run cho thay point o dau vat co offset sai nhieu, them weighting theo distance-to-centroid.

Metric nen them:

```text
offset_l1_near_centroid
offset_l1_far_from_centroid
```

### 13.4 Clustering nen tien hoa theo 2 cap

Phase instance MVP:

```text
DBSCAN(voted_centers)
```

Phase sau:

```text
DBSCAN(voted_centers + embedding)
merge/split heuristics by cluster center and point IoU
```

Pose phase:

```text
stage 1 cluster by center + rotation
stage 2 merge by center
symmetry-aware pose vote
```

### 13.5 Tiny/occluded instances phai co policy ro

Paper bo qua object co visibility thap trong mot so metric vi chung khong phai grasp target tot. Project hien tai co the lam tuong tu bang:

```text
ignore_gt_instances_below_points: 32
min_cluster_points: 64
```

Nhung van phai report:

```text
ignored_gt_instances
small_gt_instances
unmatched_visible_gt_instances
```

## 14. Diem can can than khi ap dung vao repo

1. Paper predict rotation ngay trong PS6D, nhung project hien tai nen tach phase:
   - instance segmentation truoc
   - pose regression sau
2. Paper input chi xyz, repo hien dang dung xyz + normal. Khong can bo normal neu baseline tot.
3. Paper normalize theo 100 mm workpiece cube cho multi-object dataset; repo hien chi co K41144 nen co the giu scene metric space trong instance phase.
4. Symmetry cua K41144 phai duoc xac dinh tu CAD/grasp requirement truoc khi viet rotation loss.
5. Two-stage clustering dung rotation chi nen lam sau khi quaternion distance/symmetry distance da ro.
6. Metrics instance khong du cho pose. Sau phase instance, can them pose-stage API dua tren `object_to_camera`.

## 15. De xuat update cho roadmap hien tai

Roadmap trong `docs/pointnet2-instance-segmentation-plan.md` da dung huong. Bai PS6D goi y them cac muc sau:

### Cho Phase 1-5 instance MVP

- Giu dataset wrapper sinh `centroid_offsets`.
- Them optional `centroid_distance_weights`, mac dinh off.
- Trong loss report, log offset L1 theo near/far centroid.
- Dung DBSCAN tren voted centers truoc, chua can rotation.

### Cho Phase 6-9 evaluation/preview

- Preview PLY nen to mau:
  - GT instance
  - predicted cluster
  - voted center cloud
  - unmatched/misclustered points
- Metric report:
  - predicted cluster count
  - GT visible instance count
  - ignored tiny instance count
  - merge/split count neu co Hungarian matching.

### Cho pose phase sau instance

- Them pose targets tu `metadata.instances[].object_to_camera`.
- Them model point cloud sampled tu K41144 STL.
- Implement symmetry audit truoc khi loss:
  - finite symmetry list
  - infinite symmetry axis neu co
  - grasp-equivalence decision.
- Thu nghiem pose with GT instances truoc predicted instances.
- Sau do moi lam full pipeline instance -> pose -> ICP.

## 16. Concept summary ngan

PS6D la mot pipeline point-cloud-only cho bin-picking cong nghiep. Network dung PointNet++/attention de predict centroid vote va rotation vote cho tung point. Instance segmentation va pose estimation duoc suy ra bang two-stage clustering tren cac vote nay. Bai bao dac biet tap trung vao vat dai/manh va vat doi xung, hai case rat de lam merge instance hoac sai rotation. Voi project K41144, bai nay ung ho huong PointNet++ backbone + centroid offset + clustering hien tai, dong thoi goi y them center-distance-sensitive offset loss, visibility/point-count policy, va symmetry-aware pose loss cho phase pose estimation sau nay.
