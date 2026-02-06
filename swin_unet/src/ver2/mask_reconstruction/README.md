**Mask Reconstruction**

Train the existing Swin-UNet dual-view model to predict a binary mask from a single PNG slice. Inputs are grayscale `[1,H,W]` in `[0,1]`; targets come from `*_mask.npz` stored beside each image.

**Dataset Layout**
- Folder contains pairs: `name.png` and `name_mask.npz`.
- If `--mask_key` is omitted, the first array in the NPZ is used. Nonzero values are treated as `1`.
- Image and mask are loaded at native resolution; no masking is applied to inputs.

**Dual View Behavior**
- View1 uses the raw image.
- View2 uses `flip_lr(image)`. The target for view2 is also flipped to keep spatial alignment.
- The model outputs logits `[B,1,H,W]` for both views; Dice is computed on sigmoid outputs.

**Run Command**
From `swin_unet/src/ver2`:
```
# Explicit train/val folders (preferred)
python -m mask_reconstruction.main \
  --train_dir /path/to/train_pairs \
  --val_dir /path/to/val_pairs \
  --out_dir /path/to/run_dir \
  --epochs 20 --batch_size 8 --lr 1e-3 \
  --num_workers 4 --amp 1 --strict_pairs 1

# Legacy single-folder mode (auto split; --data_dir is deprecated)
python -m mask_reconstruction.main \
  --data_dir /path/to/data \
  --out_dir /path/to/run_dir \
  --epochs 20 --batch_size 8 --lr 1e-3
```
Optional flags: `--mask_key`, `--threshold` (for dice metric binarization), `--save_best_only`, `--val_ratio` (default 0.1, only used when `--val_dir` is absent), `--cpu`.

Visualization (optional):
- `--vis-every N` enables saving validation prediction grids every N epochs (N=0 disables).
- `--vis-num K` limits how many validation samples to plot (default 4).
- `--vis-threshold T` binarizes predictions for display (default 0.5).
Outputs are saved to `out_dir/vis/val_vis_epoch_XXXX.png` showing input, target, predicted probability, thresholded mask with per-sample Dice.

**Outputs**
- `epoch_log.csv` in `out_dir` with columns: `epoch,train_loss,train_dice,val_loss,val_dice`.
- `checkpoints/latest.pt` (unless `--save_best_only 1`).
- `checkpoints/best_val_dice.pt` saved when validation Dice improves.

**Smoke Test**
Use a tiny folder with a couple of PNG/NPZ pairs:
```
python -m mask_reconstruction.main \
  --data_dir ./toy_pairs \
  --out_dir ./toy_run \
  --epochs 1 --batch_size 2 --val_ratio 0.5 --amp 0
```
Expected: command finishes, `epoch_log.csv` contains one row, and `checkpoints/best_val_dice.pt` is created.

If you have separate splits:
```
python -m mask_reconstruction.main \
  --train_dir ./toy_pairs/train \
  --val_dir ./toy_pairs/val \
  --out_dir ./toy_run \
  --epochs 1 --batch_size 2 --amp 0 --vis-every 1
```
Expected: same outputs plus `vis/val_vis_epoch_0001.png`.
