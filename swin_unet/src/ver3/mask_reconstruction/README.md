**Mask Reconstruction (ver3)**

Train Swin-UNet to reconstruct mask slices from grayscale PNG inputs.

**Sample Contract**
- `input`: float32 `[1,H,W]` in `[0,1]`
- `target`: float32 `[1,H,W]`, from mask ids as `mask/255.0` (or `(mask>0).float()` with `--binarize-target`)
- `plane_one_hot`: float32 `[2]`
- `path`: original image path string

**Raw Dataset Layout**
- Pairing rule: `name.png` with `name_mask.npz` in the same folder.
- Typical split usage:
  - `train_dir=/.../train/axial`
  - `val_dir=/.../valid/axial`

**Offline Preprocessing (recommended)**
Use this once to resize/normalize I/O offline and write preprocessed metadata:

```bash
python -m swin_unet.src.ver3.tools.preprocess_mask_dataset \
  --input-dir /data/mask/train/axial \
  --output-dir /data_preprocessed/mask/train/axial \
  --image-size 256 \
  --ext .png \
  --output-mask-suffix _mask.npy \
  --resize-mode letterbox \
  --preserve-structure \
  --num-workers 8
```

Outputs:
- resized images (`.png` by default)
- resized masks (`_mask.npy` by default)
- `preprocess_meta.json` (size, format, normalization, version)

**Train With Preprocessed Data (no online resize path)**
```bash
python -m swin_unet.src.ver3.cli train-mask \
  --preprocessed_dir /data_preprocessed/mask/train/axial \
  --val_dir /data_preprocessed/mask/valid/axial \
  --image-size 256 \
  --enable-reconstruct --disable-contrastive --disable-masking \
  --boundary-aware \
  --skip_resize_in_loader \
  --batch-size 16 --epochs 20 \
  --out-dir runs_mask --run-name mask_preprocessed
```

Notes:
- `--preprocessed_dir` auto-enables loader skip-resize behavior.
- Metadata (`preprocess_meta.json`) is validated against runtime `--image-size`.
- If metadata contains `mask_suffix`/`image_ext`, loader uses those values automatically.

**Boundary-Aware Mode**
- Enable with `--boundary-aware`.
- Behavior is always on for all epochs (no scheduling/warmup toggle).
- Boundary band is derived from the target mask contour (fixed thin band).
- Reconstruction uses weighted BCE where boundary pixels are up-weighted.
- Region-wise metrics are logged for both train and validation:
  - `dice_boundary`: Dice on boundary foreground band
  - `dice_interior`: Dice on interior foreground (`foreground - boundary`)
- Validation loss logs include:
  - `val_loss_boundary`
  - `val_loss_interior`

**Sanity Checklist**
- Folder has paired files with matching stems.
- Preprocessed `H,W` matches training `--image-size` (square expected by current model).
- Inputs are grayscale and masks remain integer id maps on disk.
- If `--binarize-target` is off, target still uses `mask/255.0` to preserve existing training semantics.
