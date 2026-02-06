**Mask Reconstruction**

Train the existing Swin-UNet dual-view model to predict a binary mask from a single PNG slice. Inputs are grayscale `[1,H,W]` in `[0,1]`; targets come from `*_mask.npz` stored beside each image.

**Dataset Layout**
- Folder contains pairs: `name.png` and `name_mask.npz`.
- If `--mask_key` is omitted, the first array in the NPZ is used. Nonzero values are treated as `1`.
- Images and masks are resized to `image_size` from the shared config (default 192, override with `--image-size`). No input pixel masking is applied.

**Dual View Behavior**
- View1 uses the raw image.
- View2 uses `flip_lr(image)`. The target for view2 is also flipped to keep spatial alignment.
- The model outputs logits `[B,1,H,W]` for both views; Dice is computed on sigmoid outputs.

**Run Command (reconstruct-only, mixed loss + dice aux)**
From `swin_unet/src/ver2`:
```
python -m mask_reconstruction.main \
  --train_dir /kaggle/input/synthstrip-data-v1-5-slices/synthstrip_data_v1.5_slices/test/axial \
  --val_dir   /kaggle/input/synthstrip-data-v1-5-slices/synthstrip_data_v1.5_slices/valid/axial \
  --out-dir after_stage0 \
  --run-name after_stage0 \
  --enable-reconstruct --disable-contrastive --dual-view \
  --epochs 10 --batch-size 32 --lr 1e-3 --image-size 256 \
  --enable_saca --saca_position after_stage0 \
  --dice-loss-weight 0.1 --dice-mode total --dice-smooth 1e-6 \
  --vis-every 2 --vis-num 4 --vis-threshold 0.5 --no-tqdm 0
```
Optional flags: `--mask_key`, `--threshold` (for dice metric binarization), `--vis-num`, `--vis-threshold`, plus all base ver2 flags (SACA, contrastive, single/dual view, checkpoints, etc.). Image and mask are resized together (bilinear for image, nearest for mask); override sizing with `--target-size` and `--resize-mode` (letterbox/direct). Enable `--debug-shapes 1` to log sample shapes.

Stability defaults: lr=3e-4, AMP enabled by default, grad clipping (--grad-clip) default 1.0, cosine LR with warmup (--warmup-epochs). Dice auxiliary enabled by default (--dice-loss-weight 0.2, --dice-mode fg). Plane conditioning set via --plane.

Visualization (optional):
- `--vis-every N` enables saving validation prediction grids every N epochs (N=0 disables).
- `--vis-num K` limits how many validation samples to plot (default 4).
- `--vis-threshold T` binarizes predictions for display (default 0.5).
Outputs are saved to `out_dir/vis/val_vis_epoch_XXXX.png` showing input, target, predicted probability, thresholded mask with per-sample Dice.

**Outputs**
- `epoch_log.csv` in `out_dir` with columns: epoch, train/val loss_total, loss_masked, loss_unmasked, dice, train_loss_dice_aux, val_loss_dice_aux, train_loss_contrastive, val_loss_contrastive, lr.
- Plots in `out_dir/plot/`: `loss_total.png`, `loss_masked.png`, `loss_unmasked.png`, `dice.png`, `dice_aux.png`, `contrastive.png`.
- `checkpoints/latest.pt` (unless `--save_best_only 1`).
- `checkpoints/best_val_dice.pt` saved when validation Dice improves.

**Smoke Test**
Use a tiny folder with a couple of PNG/NPZ pairs:
```
python -m mask_reconstruction.main \
  --data-root ./toy_pairs \
  --train_dir ./toy_pairs \
  --out-dir ./toy_run \
  --epochs 1 --batch_size 2 --val_ratio 0.5 --amp 0
```
Expected: command finishes, `epoch_log.csv` contains one row, and `checkpoints/best_val_dice.pt` is created.

If you have separate splits:
```
python -m mask_reconstruction.main \
  --train_dir ./toy_pairs/train \
  --val_dir ./toy_pairs/val \
  --out-dir ./toy_run \
  --epochs 1 --batch_size 2 --amp 0 --vis-every 1
```
Expected: same outputs plus `vis/val_vis_epoch_0001.png`.
