
- Thử lại patch + stage 0
- Tất cả vị trí
- T1 axial 
- So sánh SACA:
    Không SACA
    1 vị trí có SACA (4 version)
    Combine (Patched emb + Stage 0, tất cả 4 vị trí) (2 version)
==> tổng 7 => tìm ra phiên bản tốt nhất 

- Dùng best model pretrained 
=> thử recontruct mask / classi (mindset) 

- 512/384 bottleneck classification (huggingface), classifier deeper (3 hidden layer :2 dim mỗi lần)

- SynthStrip cắt ảnh lấy 5 lát 


Áp dụng trực tiếp model:
- Refine lại attention: cross attention trước swin (trực tiếp trên token) (bật/tắt)
- Thử lại contrastive (vị trí đặt, loại loss)
- Sau khi cross attention: pretrained encoder: finetune
    + Segmentation (SynthStrip: skull skipping / tissue segmentation (check lại task này) [decode ra trực tiếp masking] - đổi lại output cho decode - reuse pretrain weight encoder only, init from stratch for decoder, use same architecture - bỏ phần mask có chiến lược, dùng full image, output decode ra mask) (thử trước 2D - sagittal)
    + Segmentation for Brain Tumor (y hệt như phần Skull Stripping) (? dataset)
    + Classification (s2_1 & s2_2 concat (fusion, ...) + MLP (linear)) + (imbalanced handling): baseline first -  focal loss vs weighted loss, ... : mindset dataset (AD vs non-AD | abnormalities) , hgf dataset (sắp xếp lại nhãn: 0-4 nhẹ đến nặng) phân lớp bth 4 nhãn, lấy nhãn sắp xếp thứ tự 0-4 theo nặng nhẹ, normalize, activation sigmoid, ce loss/rmse/ranking loss, ...  Phân lớp loại Tumor (? dataset)
    

- 600 bn khỏe, independent vào loại bệnh

https://www.kaggle.com/datasets/hoanggvo/free-surfer

https://www.kaggle.com/datasets/briscdataset/brisc2025/data

mindset

huggingface

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset



| **Dataset**                                             | **Loại ảnh & Mặt phẳng**                                                                                  | **Tác vụ chính**                                                                 | **Nhãn cung cấp**                                                                                                                                            | **Quy mô (số mẫu) & Định dạng**                                                                                           | **Truy cập**                                                                                                              | **Tài liệu tham khảo chính**                                                                                   |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **BraTS** (Multimodal Brain Tumor Segmentation)         | 3D MRI đa chuỗi (T1, T1Gd, T2, FLAIR), <br>độ phân giải 1mm, mặt phẳng axial (gốc).                       | Segmentation khối u não (glioma).                                                | Mask phân đoạn vùng **u não**: lõi u (necrotic), phù, vùng tăng quang (enhancing).<br>*(Có nhãn phụ: phân loại HGG/LGG, OS trong một số năm).*               | ~369 ca train + 125 ca val (BraTS’20);<br>mỗi ca gồm 4 volume MRI + 1 volume nhãn (NIfTI, 240×240×155).                   | Đăng ký trên site BraTS (CBICA) để tải;<br>hoặc tải bản Kaggle (BraTS2020, BraTS2021) – miễn phí.                         | Menze *et al.*, TMI 2015;<br>Bakas *et al.*, 2018 (ArXiv) – BraTS tổng kết.                                    |
| **MRBrainS** (Brain Tissue Segmentation Challenge 2013) | 3D MRI **3 chuỗi** (T1, T1-IR, T2-FLAIR) chụp 3T; <br>ảnh đã co-registered, độ phân giải ~0.95×0.95×3 mm. | Segmentation **chất xám / trắng / dịch** (não lành).                             | Mask phân đoạn **3 mô não**: GM, WM, CSF (kèm nền) trên toàn não.                                                                                            | 20 case (5 train + 15 test);<br>mỗi case có 3 volume MRI + 1 volume nhãn (NIfTI/MHA, ~240×240×48).                        | Cần đăng ký và ký thỏa thuận trên mrbrains13.isi.uu.nl;<br>dữ liệu miễn phí cho nghiên cứu, hạn chế dùng ngoài challenge. | Mendrik *et al.*, CIN 2015 – MRBrainS13.                                                                       |
| **IBSR** (Internet Brain Segmentation Repository)       | 3D MRI T1-weighted (1.5T),<br>góc chụp axial; có một số T2.                                               | Segmentation **cấu trúc não** (đa năng: mô, vùng dưới vỏ, skull stripping).      | **Phân vùng giải phẫu** (≈ 40+ labels: não thất, hồi hải mã…);<br>**Mask não** (binary) cho mỗi ảnh.<br>*Có thể gom thành GM/WM/CSF để đánh giá thuật toán*. | 18 ảnh MRI có nhãn (IBSR v2) + 20 ảnh thường khác;<br>định dạng Analyze 7.5 và NIfTI (ảnh ~256³); license phi thương mại. | Tải trực tiếp từ NITRC (proj IBSR);<br>không cần đăng ký, dùng phi lợi nhuận.                                             | NITRC IBSR summary;<br>McDonald *et al.*, 2017 (NIH) – ứng dụng IBSR.                                          |
| **SynthStrip** (Skull-Stripping Dataset, 2022)          | 3D **đa modal** (MRI, CT, PET) full head; <br>+ 2D lát giữa (sagittal) cho mỗi volume.                    | **Skull stripping** (tách mô não); <br>phân đoạn giải phẫu (trên MRI con người). | **Mask não nhị phân** cho tất cả 622 scans;<br>**Label giải phẫu FreeSurfer** cho 131 MRI (80+ vùng, gồm non-brain).                                         | 622 scans (≈131 MRI + 491 CT/PET) – 6.9 GB;<br>+ 622 ảnh 2D (.png) – 39 MB. Định dạng NIfTI (.nii.gz).                    | Mở tải về trên trang FreeSurfer (MIT/CC license);<br>không cần đăng ký, trích dẫn nếu sử dụng.                            | Hoopes *et al.*, NeuroImage 2022 – giới thiệu SynthStrip.<br>Kelley *et al.*, ISBI 2024 – ứng dụng SynthStrip. |
| **Figshare Brain Tumor** (MRI U Não 3 Lớp)              | 2D MRI T1 CE (đối quang), lát axial/coronal/sagittal; <br>resolution 256×256 hoặc 512×512.                | **Classification đa lớp** – phân loại loại u từ ảnh MRI.                         | **Label lớp u**: *Glioma*, *Meningioma*, *Pituitary tumor* (3 lớp) cho mỗi ảnh.                                                                              | 3064 ảnh từ 233 bệnh nhân (G:1426, M:708, P:930);<br>định dạng .jpg/.png, ~8-bit.                                         | Mở trên Figshare (không cần account);<br>link DOI do Cheng *et al.* cung cấp.                                             | Cheng *et al.*, PLoS ONE 2015;<br>Afshar *et al.*, Nature Sci Rep 2020 – sử dụng dataset.                      |
| **Kaggle Brain MRI 4-class** (Tumor vs Normal)          | 2D MRI T1 (nhiều CE), lát axial/coronal/sagittal;<br>ảnh đã chuẩn kích thước (thường 64–512px).           | **Classification 4 lớp** – nhận biết ảnh **có u hay không**, và loại u.          | **Label 4 lớp**: *No Tumor*, *Glioma*, *Meningioma*, *Pituitary*.                                                                                            | 7023 ảnh (train/test) – 4 lớp cân đối ~1.7k ảnh/lớp;<br>.png/.jpg (8-bit).                                                | Tải từ Kaggle Datasets (yêu cầu login);<br>mục đích phi thương mại, hoàn toàn miễn phí.                                   | AbdElHamid *et al.*, Diagnostics 2023 – dùng dataset;<br>Pacal *et al.*, BrainRes 2025 – SoTA 99.78%.          |
