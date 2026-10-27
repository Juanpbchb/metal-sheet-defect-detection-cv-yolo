# Undergraduate Research — IMUSA: Visual Defect Inspection of Metal Sheets/Pans with Computer Vision

Undergraduate research project (Universidad Nacional de Colombia, 2024-02) carried out for **IMUSA**. The goal is to assess the feasibility of a computer vision system to automate quality control of metal sheets (frying-pan discs), detecting defects such as **scratches, dents, discontinuities in the engraving, and missing/defective thermo-signal**.

The work combines two complementary approaches:

1. **Classical image processing** (OpenCV / scikit-image) to enhance and segment engraving defects.
2. **Deep learning detection** (Ultralytics YOLO) trained on raw images with data augmentation.

---

## Repository structure

```
.
├── presentacion.pdf                   # Project presentation: objectives, dataset, filters, results and conclusions
├── preprocesamienti-imagenes.ipynb    # Exploratory analysis and classical defect enhancement/segmentation pipeline
├── yolo-implementacion.ipynb          # Data augmentation, auto-labeling, split and YOLOv8 training
├── resultado-yolo.mp4                 # Demo video of model inference
└── runs/content/runs/detect/
    └── sarten_defectos_y_sin_defectos/
        ├── args.yaml                  # Full training hyperparameters
        ├── results.csv                # Per-epoch metrics (losses, precision, recall, mAP)
        ├── events.out.tfevents.*      # TensorBoard logs
        └── weights/
            ├── best.pt                # Best checkpoint (by validation fitness)
            ├── last.pt                # Last checkpoint
            ├── epoch0.pt / epoch10.pt # Periodic checkpoints (save_period=10)
```

> [!NOTE]
> The image dataset is not versioned in this repository. The notebooks expect a `Dataset/` folder (preprocessing) and an `images.zip` with `defects/` and `no_defect/` subfolders (YOLO). They were run on Deepnote and Google Colab respectively. Notebook contents are written in Spanish.

---

## 1. Dataset acquisition

- **114 images**: 40 good sheets and 74 defective ones (scratches, dents, missing thermo-signal, etc.).
- Captured in an **acoustically isolated room with anti-reflective walls and total darkness** to control illumination.
- **Diagonal lighting of ~500 lm**, camera at **~20 cm**, fixed camera parameters (focal length, exposure; see EXIF) and a **dark background** to avoid saturation.
- File naming convention: `A_B_C_D.jpg`
  - `A`: state (`B` good / `M` bad)
  - `B`: sheet size (`P` small / `G` large)
  - `C`: sheet id within the category
  - `D`: shot id of that sheet

**Data challenges identified**

| Issue | Description |
|---|---|
| Sensitivity | Defects such as dents are barely visible to the naked eye. |
| Angle dependency | Visibility of some defects depends strongly on lighting/capture angle. |
| Class imbalance | More defective than good samples → data augmentation required. |

Defects were grouped into two families: **engraving defects** and **thermo-signal defects**. Classical preprocessing focused on engraving defects.

---

## 2. Classical image preprocessing (`preprocesamienti-imagenes.ipynb`)

Libraries: `opencv-python`, `numpy`, `scikit-image`, `matplotlib`.

Explored pipeline:

1. **Histogram analysis** (grayscale and RGB) of the reference image: bimodal separation between background and sheet.
2. **Conversion to CIELAB** and channel split (`L`, `a`, `b`). The `a` channel isolates the thermo-signal well (reddish component), while `L` carries most of the engraving information.
3. **Frequency-domain high-pass filter**: 2D FFT (`np.fft.fft2` + `fftshift`), circular mask suppressing low frequencies (r = 30) and IFFT to recover edges and fine textures (scratches, discontinuities).
4. **Division of normalized images** (`filtered / (original + ε)`) to compensate for non-uniform illumination and highlight local defect contrast.
5. **Otsu thresholding** to binarize and remove background and thermo-print noise.
6. **Segmentation mask** (`bitwise_and`) and semi-transparent **overlay** on the original image for visual inspection.
7. **CLAHE** (Contrast Limited Adaptive Histogram Equalization): the mask built from the good sheet failed on defective images; improving local contrast with CLAHE fixed the issue.
8. Variants described in the presentation: selection and **sum of color channels**, **negative**, and **intensity boost** to remove the thermo-signal and keep only the engraving.

Results: the mask correctly enhances small scratches, pronounced scratches, small dark scratches and light engraving defects.

### Thermo-signal isolation and feature matching

- Thermo-signal segmentation using the LAB `a` channel (shifted by `-150`) + Otsu + `morphology.remove_small_objects` (scikit-image), then bounding-box cropping.
- **Feature matching with SIFT + BFMatcher**, top-10 matches and **RANSAC homography** to locate the template in the image (ORB was tested and discarded).
- Works on the same reference image but **generalizes poorly** to other sheets; left as an open line of work.

---

## 3. YOLO detection (`yolo-implementacion.ipynb`)

Libraries: `ultralytics`, `albumentations`, `opencv-python`, `scikit-learn`, `wandb`.

### 3.1 Data augmentation and balancing
Using `albumentations`, ~**1000 images per class** (`defectuosa` = defective, `sin_defectos` = defect-free) are generated:
`RandomRotate90`, `HorizontalFlip`, `VerticalFlip` (p=0.5), `GaussNoise`, `GaussianBlur(3–7)` (p=0.3).

### 3.2 Auto-labeling
YOLO-format labels are generated automatically with OpenCV: grayscale → Gaussian blur → inverted Otsu → morphological closing → **largest external contour** → normalized `boundingRect`. Each image gets one bounding box covering the whole sheet, labeled with its folder's class. The presentation also mentions **Roboflow** for defect labeling.

### 3.3 Dataset organization
Standard `dataset/images/{train,val}` and `dataset/labels/{train,val}` layout with an **80/20** split (`train_test_split`, `random_state=42`) and `data.yaml`:

```yaml
train: /content/dataset/images/train
val: /content/dataset/images/val
names:
  0: defectuosa
  1: sin_defectos
```

### 3.4 Training
- Base model: pretrained **YOLOv8n** (`yolov8n.pt`); the presentation mentions YOLOv11 as the reference version.
- `imgsz=640`, `batch=16`, `epochs=20` (per `args.yaml`), `patience=15`, `lr0=0.001`, `lrf=0.01`, `warmup_epochs=3`, `auto` optimizer, AMP enabled.

### 3.5 Results (`results.csv`, last epoch)

| Metric | Value |
|---|---|
| Precision (B) | 0.9997 |
| Recall (B) | 1.000 |
| mAP@50 | 0.995 |
| mAP@50-95 | 0.995 |

> [!WARNING]
> These metrics should be read with caution: (a) the split is performed **after** augmentation, so variants of the same physical sheet can end up in both train and val (*data leakage*); (b) bounding boxes cover the whole sheet, so the task behaves as **image-level classification** rather than defect localization. Consistently, the presentation concludes that **YOLO on raw images was not very effective** in practice.

---

## 4. Conclusions and future work

- Classical filters (LAB + high-pass + division + Otsu + CLAHE) work very well for **scratches and engraving discontinuities**.
- YOLO trained on unprocessed images did not generalize adequately.
- Next steps:
  - Train YOLO on **images previously passed through the designed mask**.
  - Annotate defects at **per-defect bounding box/segmentation** level and split **by physical sheet** before augmentation.
  - Re-evaluate the system in the **real production-line setting**.

---

## Quick start

```bash
pip install opencv-python scikit-image matplotlib albumentations ultralytics scikit-learn
```

Inference with the trained model:

```python
from ultralytics import YOLO
model = YOLO("runs/content/runs/detect/sarten_defectos_y_sin_defectos/weights/best.pt")
model.predict("path/to/image.jpg", save=True)
```

## References

- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2015). *You Only Look Once: Unified, Real-Time Object Detection*. arXiv:1506.02640.
- Roboflow — https://roboflow.com/
- OpenCV docs — https://docs.opencv.org/4.x/
- Ultralytics YOLO — https://docs.ultralytics.com/
