# Instance Segmentation Based 6D Pose Estimation For Bin Picking

Nguon: Chungang Zhuang, Shaofei Li, Han Ding, "Instance segmentation based 6D pose estimation of industrial objects using point clouds for robotic bin-picking", Robotics and Computer-Integrated Manufacturing 82, 2023.

## 1. Muc tieu bai bao

Bai bao de xuat mot pipeline deep learning de uoc luong 6D pose cho vat the cong nghiep trong bai toan robotic bin picking. Doi tuong muc tieu la cac vat it texture hoac khong co texture, bi xep chong, bi che khuat, va nam trong scene lon.

Thay vi dua vao RGB, bai bao dung point cloud lam input chinh vi point cloud:

- Bieu dien truc tiep hinh hoc 3D cua vat.
- It bi anh huong boi dieu kien anh sang.
- Phu hop hon voi vat cong nghiep it dac trung mau-sac/texture.

Ket qua mong muon cua he thong la tim tung object instance trong scene, uoc luong pose cua tung instance, refine pose bang ICP, roi chon object phu hop de robot gap.

## 2. Y tuong cot loi

Pipeline gom 2 stage:

1. Instance segmentation tren scene point cloud.
2. 6D pose estimation cho tung instance point cloud.

Cach chia hai stage giup pose network khong phai xu ly ca scene lon voi nhieu vat che khuat lan nhau. Sau segmentation, moi lan pose estimation chi can xu ly mot object instance, nen bai toan nho hon, ro hon va nhanh hon.

Sau khi co pose du doan, bai bao dung ICP de refine pose lan cuoi truoc khi robot gap.

## 3. Input va output

Input tong the:

- Scene point cloud tu 3D sensor/depth camera.

Output cua instance segmentation:

- Semantic label cho moi point: point thuoc loai vat nao.
- Instance embedding cho moi point: dung de gom cac point thanh tung object rieng.
- Instance point cloud: point cloud da tach rieng cho tung vat.

Output cua pose estimation:

- Translation vector.
- Rotation quaternion.
- 6D pose/rigid transformation cua object so voi object model/template.

## 4. Synthetic dataset bang physical simulation

Mot dong gop quan trong cua bai bao la dung physical simulation de tao dataset thay vi annotate point cloud thu cong.

Quy trinh dataset:

1. Scan vat that bang 3D scanner tu nhieu view.
2. Dang ky cac scan de tao complete point cloud cua object.
3. Dung ball pivoting de tai tao surface va tao geometric model.
4. Dua nhieu object model vao Blender/Bullet simulation.
5. Tha vat ngau nhien theo gravity, co collision detection, de tao cluttered scene.
6. Khi scene on dinh, lay pose ground truth cua tung object.
7. Render depth map theo camera/sensor parameters.
8. Chuyen depth map thanh point cloud.
9. Tao semantic label, instance label va pose label.

Dataset synthetic duoc dung cho ca:

- Training instance segmentation.
- Training pose estimation.

Ly do cach nay quan trong cho project bin picking: annotation pose/instance cua point cloud that rat ton cong, trong khi simulation co the sinh nhieu scene da dang voi label tu dong.

## 5. Instance segmentation network

### 5.1 Van de can giai quyet

Point cloud scene trong bin picking co nhieu object xep chong va che khuat. Neu sample scene xuong qua it point, moi object co the mat nhieu thong tin hinh hoc. Neu dung similarity matrix n x n giua moi cap point nhu mot so method truoc, memory se tang rat manh khi so point lon.

Bai bao thiet ke instance segmentation de xu ly point cloud co nhieu point hon ma khong tao ma tran n x n.

### 5.2 Kien truc chinh

Backbone:

- PointNet++ de extract feature tu point cloud.

Network tach thanh cac branch:

- Semantic branch: hoc semantic feature va predict class cua moi point.
- Instance branch: hoc instance embedding cua moi point.
- Distance feature branch: hoc feature tu toa do 3D de giup phan biet cac object cung class nhung nam o vi tri khac nhau.

Fusion:

- Semantic feature duoc fuse vao instance feature de day xa embedding cua cac class khac nhau.
- Distance feature duoc fuse tiep vao instance feature de tach cac instance cung class.

Sau do, instance embedding duoc cluster de tao instance label.

### 5.3 Loss

Semantic branch:

- Cross entropy loss.

Instance branch:

- Discriminative loss gom 3 thanh phan:
  - Variance term: keo embedding cua point trong cung instance lai gan nhau.
  - Distance term: day mean embedding cua cac instance khac nhau ra xa nhau.
  - Regularization term: giu embedding khong tro nen qua lon.

### 5.4 Clustering sau network

Quy trinh clustering:

1. Mean-shift clustering tren instance embeddings.
2. DBSCAN clustering lan hai de loai point nhieu/noise va tach instance tot hon.
3. Chon cluster co nhieu point nhat lam instance output cuoi.
4. Semantic class cua instance lay theo mode cua semantic labels trong cluster.

Bai bao dung DBSCAN o buoc sau vi DBSCAN cham hon, nhung phu hop de loai cac point roi rac sau khi so point da giam.

## 6. Pose estimation network

### 6.1 Vi sao khong dua point cloud thang vao pose network?

Moi instance sau segmentation co so point khac nhau. Sampling ve cung so point co the lam mat thong tin hinh hoc. Bai bao chon projection point cloud sang feature map 2D de tao structured input cho CNN.

### 6.2 Feature generation

Tu instance point cloud, tao hai loai feature map:

- Depth feature map.
- Normal feature map.

Depth map giu thong tin hinh dang theo do sau. Normal map bo sung thong tin be mat, giup phan biet cac pose/object co depth projection giong nhau.

Feature map duoc tao bang cach:

1. Dua instance point cloud ve goc toa do.
2. Xac dinh mien toa do theo toan bo training set.
3. Project point len mat phang 2D.
4. Moi pixel luu point gan sensor/origin hon.
5. Tinh normal tu neighborhood cua point va project tuong tu.

### 6.3 Kien truc pose network

Network co hai branch CNN:

- Depth branch: ResNet101 pretrained + fully connected layers.
- Normal branch: ResNet50 + fully connected layers.

Feature tu hai branch duoc concatenate, sau do tach thanh:

- Rotation branch: predict quaternion.
- Translation branch: predict translation.

Rotation duoc bieu dien bang quaternion.

### 6.4 Xu ly symmetric object

Vat cong nghiep thuong co doi xung. Cung mot trang thai vat ly co the tuong ung voi nhieu pose matrix hop le. Neu train bang ADD loss thong thuong, network co the bi phat sai vi du doan mot pose tuong duong nhung khac ground truth label.

Bai bao de xuat weighted ADD loss voi multi-channel output:

- Network khong predict mot pose duy nhat, ma predict nhieu pose channel.
- Voi symmetric object co k pose mo ho, network output k pose.
- Channel co ADD nho nhat voi ground truth duoc gan trong so lon.
- Cac channel con lai van duoc train voi trong so nho hon.

Cach nay giup network hoc cac pose mo ho cua vat doi xung ma khong can loai bo sample khoi dataset.

## 7. ICP refinement

Pose network tao rough pose/initial pose. Sau do ICP duoc dung de refine alignment giua:

- Model point cloud da transform boi pose du doan.
- Instance point cloud trong scene.

ICP giup tang do chinh xac cuoi, nhung can initial pose tot de tranh roi vao local optimum. Vi vay deep network dong vai tro tao initial pose du manh.

## 8. Robotic bin picking pipeline

Quy trinh thuc te:

1. 3D scanner chup scene va tao point cloud.
2. Instance segmentation tach cac object trong bin.
3. Loc va cluster lai instance point cloud.
4. Pose estimation cho cac instance ung vien.
5. ICP refinement.
6. Chon object nam tren cung dua vao mean z cua point cloud.
7. Neu pose dat dieu kien, robot gap object do.
8. Sau moi lan gap, scan lai scene va lap lai.

He robot trong bai bao:

- UR5 robot.
- PhoXi 3D scanner.
- Gripper duoc thiet ke rieng de gap nhieu object cong nghiep.

Quan he toa do:

- Sensor frame -> robot base frame: lay tu hand-eye calibration.
- Sensor frame -> object frame: pose network predict.
- Object frame -> gripper frame: gan co dinh theo grasp pose da define.
- Gripper frame -> robot base frame: gui cho robot controller.

## 9. Thuc nghiem va ket qua chinh

### 9.1 Instance segmentation

Dataset public S3DIS:

- Method de xuat dat semantic segmentation tot hon ASIS va 3D-BoNet ve mAcc, mIoU va oAcc.
- Instance segmentation co mean precision nhinh hon ASIS va cao hon ro ret 3D-BoNet.

Synthetic bin-picking dataset:

- 15,000 scene samples.
- Moi sample co 16,384 points.
- Train/validation = 8/2.
- Instance segmentation dat mean precision khoang 92.6% va mean recall khoang 95.9% voi IoU threshold 0.9.

Real dataset:

- 100 real scene samples.
- Moi sample co 16,384 points.
- Mean precision giam xuong khoang 68.0% do noise va reality gap.
- Mean recall van cao, khoang 94.4%, phu hop voi bin picking vi robot chu yeu can tim object co the gap o lop tren.

### 9.2 Pose estimation

Pose estimation duoc so sanh voi SAC-IA va PPF, ca ba deu refine bang ICP.

Ket qua tren synthetic dataset:

- SAC-IA kem hon dang ke.
- PPF rat manh voi mot so object, nhung yeu voi hammer do doi xung hinh hoc gay nham mat truoc/sau.
- Method de xuat on dinh hon giua cac object, dat accuracy khoang 94-97% tuy object.

Inference time:

- Method de xuat nhanh hon SAC-IA ro ret.
- Nhanh hon PPF mot chut.
- Single forward time cua full network khoang 100-300 ms.

### 9.3 Bin picking

Trong thi nghiem robot:

- Robot gap thanh cong voi tung loai object rieng va voi cac combination nhieu loai object.
- Grasping test co success rate 100% trong setup cua bai bao.
- Average cycle time khoang 48.2 s, chu yeu ton thoi gian o acquisition va robot motion, khong phai inference.

## 10. Han che duoc tac gia neu

1. Reality gap giua synthetic va real point cloud lam giam pose accuracy khi deploy that.
2. Sensor dat co dinh nen co vat o lop tren nhung van bi scan thieu thong tin hinh hoc.
3. Cac vat bi che khuat manh co the khong du point de segmentation/pose estimation tot.
4. Picking strategy chon object tren cung giup tranh va cham voi mieng bin, nhung chua giai quyet day du cho bin sau trong ung dung cong nghiep.
5. Parameter eta trong weighted ADD loss duoc chon theo kinh nghiem, chua co phan tich ly thuyet/ablation that chat.
6. Tac gia goi y future work co the ket hop bandwidth computation va confidence learning module tu PPR-Net++ de cai thien pose estimation.

## 11. Diem can hoc de implement project cua minh

Neu xay model ung dung cho bin picking tu bai nay, co the chia project thanh cac module sau:

### Module 1: Object model preparation

- Thu thap CAD model hoac scan object.
- Chuan hoa object coordinate frame.
- Tao model point cloud/surface mesh.
- Define grasp pose cho tung object.

### Module 2: Synthetic scene generator

- Dung physics simulation de tha random object vao bin.
- Random object category, count, initial pose.
- Render depth theo camera intrinsics/extrinsics.
- Convert depth thanh point cloud.
- Export labels:
  - semantic label per point.
  - instance label per point.
  - 6D pose per object.

### Module 3: Point cloud preprocessing

- Crop ROI cua bin.
- Remove background/table/bin neu can.
- Downsample hoac sample ve so point phu hop.
- Normalize toa do theo convention cua model.
- Tinh normal neu dung normal feature map.

### Module 4: Instance segmentation

- Backbone nen bat dau voi PointNet++ hoac mot point cloud backbone hien dai hon.
- Output:
  - semantic logits per point.
  - instance embedding per point.
  - optional distance/offset feature.
- Loss:
  - cross entropy cho semantic.
  - discriminative embedding loss cho instance.
- Postprocess:
  - mean-shift/DBSCAN clustering.
  - filter cluster qua nho.
  - lay semantic class theo voting.

### Module 5: Pose estimation

- Voi moi instance:
  - project thanh depth feature map.
  - tinh va project normal feature map.
- CNN hai branch:
  - depth branch.
  - normal branch.
- Predict:
  - quaternion.
  - translation.
- Dung loss phu hop voi asymmetric/symmetric object.

### Module 6: Pose refinement

- Dung ICP de refine pose.
- Can reject pose neu ICP residual/error qua lon.
- Co the dung threshold theo object type.

### Module 7: Picking policy

- Chon object o lop tren: mean/max z cao.
- Kiem tra collision/grasp feasibility.
- Chon grasp pose da define theo object.
- Transform pose sang robot frame bang hand-eye calibration.
- Sau moi pick, scan lai scene.

## 12. Huong implement thuc dung nen uu tien

Voi project cua minh, nen bat dau tu mot MVP nhu sau:

1. Chi ho tro 1-2 loai object truoc.
2. Tao synthetic dataset tu CAD/mesh va depth renderer.
3. Train instance segmentation rieng.
4. Ban dau co the thay pose network bang PPF/ICP baseline de co robot pipeline hoat dong.
5. Sau khi segmentation on, implement pose network depth+normal projection.
6. Them symmetric loss khi gap vat doi xung.
7. Lam evaluation synthetic -> real de do reality gap.

Ly do: bin picking la system problem, neu implement tat ca model cung luc se kho debug. Tach segmentation, pose, ICP va robot policy thanh cac module doc lap se giup do loi nhanh hon.

## 13. Concept summary ngan

Bai bao xem bin picking nhu bai toan "tach vat truoc, uoc luong pose sau". Scene point cloud duoc instance segmentation de lay tung object rieng. Moi object duoc project thanh depth/normal feature maps, dua qua CNN de predict quaternion va translation. Symmetry duoc xu ly bang multi-channel weighted pose loss. Synthetic physics simulation duoc dung de tao dataset co label day du. ICP refine pose cuoi. Robot chon object tren cung, gap, roi scan lai scene.
