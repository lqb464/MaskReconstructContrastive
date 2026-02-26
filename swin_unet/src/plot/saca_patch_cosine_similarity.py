from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, get_args, get_origin, get_type_hints

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from swin_unet.src.ver3.config.experiment import (
    ExperimentConfig,
    build_argparser as build_ssl_argparser,
)
from swin_unet.src.ver3.data.dataset import plane_to_one_hot
from swin_unet.src.ver3.models.model_utils import flip_lr, flip_lr_nhwc
from swin_unet.src.ver3.models.swin_unet_dualview_ssl import SwinUNetDualViewSSL
from swin_unet.src.ver3.training.ckpt_io import (
    load_checkpoint_weights,
    load_checkpoint_weights_filtered,
)
from swin_unet.src.ver3.training.utils import get_device


SACA_POINTS = ("after_patch_embed", "after_stage0", "after_merge0", "after_stage1")


def _dataclass_from_dict(dc_type, raw: dict):
    if not is_dataclass(dc_type):
        raise TypeError(f"{dc_type} is not a dataclass")

    type_hints = get_type_hints(dc_type)
    kwargs = {}
    for f in fields(dc_type):
        name = f.name
        if name not in raw:
            continue
        val = raw[name]
        ftype = type_hints.get(name, f.type)

        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[name] = _dataclass_from_dict(ftype, val)
            continue

        origin = get_origin(ftype)
        args = get_args(ftype)
        if origin is Union and isinstance(val, dict):
            dc_candidates = [a for a in args if is_dataclass(a)]
            if dc_candidates:
                kwargs[name] = _dataclass_from_dict(dc_candidates[0], val)
                continue

        kwargs[name] = val
    return dc_type(**kwargs)


def _cfg_from_ckpt(ckpt_path: Path) -> ExperimentConfig:
    obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid checkpoint format at {ckpt_path}.")
    raw_cfg = obj.get("cfg", None)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Checkpoint {ckpt_path} has no 'cfg' dictionary.")
    return _dataclass_from_dict(ExperimentConfig, raw_cfg)


def _build_model(cfg: ExperimentConfig, device: torch.device) -> SwinUNetDualViewSSL:
    return SwinUNetDualViewSSL(
        in_ch=cfg.model.in_ch,
        image_size=cfg.data.image_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        enc_depths=tuple(cfg.model.enc_depths),
        dec_depths=tuple(cfg.model.dec_depths),
        num_heads=tuple(cfg.model.num_heads),
        window_size=cfg.model.window_size,
        proj_dim=cfg.model.proj_dim,
        plane_inject_method=cfg.model.plane_inject_method,
        enable_saca=cfg.model.enable_saca,
        saca_position=cfg.model.saca_position,
        saca_positions=list(cfg.model.saca_positions),
        saca_gate_init=cfg.model.saca_gate_init,
        saca_warmup_epochs=cfg.model.saca_warmup_epochs,
        enable_reconstruct=cfg.training.enable_reconstruct,
        enable_contrastive=cfg.training.enable_contrastive,
        contrastive_loss_type=cfg.contrast_loss.contrastive_loss_type,
        contrastive_position=cfg.contrast_loss.contrastive_position,
        single_view=cfg.training.single_view,
    ).to(device)


def _load_encoder_state_filtered_compat(
    *,
    ckpt_path: Path,
    model: SwinUNetDualViewSSL,
    device: torch.device,
) -> None:
    obj = torch.load(ckpt_path, map_location=device)
    if not isinstance(obj, dict) or "model" not in obj or (not isinstance(obj["model"], dict)):
        raise ValueError(f"Invalid checkpoint format at {ckpt_path}. Expected dict with key 'model'.")

    encoder_prefixes = model.encoder_state_dict_prefixes()
    exclude_prefixes = ("proj_c1", "proj_c2", "proj_c3", "proj")
    raw_sd = {
        k: v
        for k, v in obj["model"].items()
        if str(k).startswith(encoder_prefixes) and (not str(k).startswith(exclude_prefixes))
    }

    model_sd = model.state_dict()
    filtered_sd = {}
    dropped_shape = []
    converted_gate = 0

    for k, v in raw_sd.items():
        if k not in model_sd:
            continue
        target_v = model_sd[k]
        src_v = v

        if (
            str(k).endswith(".gate")
            and torch.is_tensor(src_v)
            and torch.is_tensor(target_v)
            and target_v.ndim == 1
            and src_v.numel() == 1
        ):
            src_v = src_v.reshape(1).to(dtype=target_v.dtype, device=target_v.device).repeat(target_v.numel())
            converted_gate += 1

        if tuple(src_v.shape) != tuple(target_v.shape):
            dropped_shape.append((k, tuple(src_v.shape), tuple(target_v.shape)))
            continue

        filtered_sd[k] = src_v

    msg = model.load_state_dict(filtered_sd, strict=False)
    print(
        f"[ckpt] {ckpt_path.name}: encoder_only compatibility load | "
        f"loaded={len(filtered_sd)} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} "
        f"gate_converted={converted_gate} dropped_shape={len(dropped_shape)}"
    )


def _load_weights(
    *,
    model: SwinUNetDualViewSSL,
    ckpt_path: Path,
    ckpt_load_mode: str,
    device: torch.device,
) -> None:
    mode = str(ckpt_load_mode or "encoder_only").lower().strip()
    if mode == "none":
        print("[ckpt] ckpt_load_mode=none -> fallback to encoder_only for plotting.")
        mode = "encoder_only"

    if mode == "full":
        load_checkpoint_weights(
            ckpt_path=ckpt_path,
            device=device,
            model=model,
            strict=True,
        )
        print(f"[ckpt] {ckpt_path.name}: full load done (strict=True)")
        return

    if mode != "encoder_only":
        raise ValueError(f"Unsupported ckpt_load_mode: {ckpt_load_mode}")

    try:
        obj = load_checkpoint_weights_filtered(
            ckpt_path=ckpt_path,
            device=device,
            model=model,
            include_prefixes=model.encoder_state_dict_prefixes(),
            exclude_prefixes=("proj_c1", "proj_c2", "proj_c3", "proj"),
        )
        msg = obj.get("_load_msg", {}) if isinstance(obj, dict) else {}
        print(
            f"[ckpt] {ckpt_path.name}: encoder_only load done | "
            f"missing={len(msg.get('missing_keys', []))} unexpected={len(msg.get('unexpected_keys', []))}"
        )
    except RuntimeError as e:
        print(f"[ckpt] encoder_only native load failed, fallback compatibility loader. reason={e}")
        _load_encoder_state_filtered_compat(
            ckpt_path=ckpt_path,
            model=model,
            device=device,
        )


def _load_gray_tensor(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("L")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return x.to(device=device, dtype=torch.float32)


def _parse_plane(plane_arg: str, image_path: Path, cfg_plane: str) -> str:
    p = str(plane_arg).strip().lower()
    if p in {"axial", "coronal"}:
        return p
    if p == "auto":
        s = str(image_path).lower()
        if "coronal" in s:
            return "coronal"
        return "axial"
    cfg_p = str(cfg_plane).strip().lower()
    if cfg_p in {"axial", "coronal"}:
        return cfg_p
    return "axial"


def _extract_position_representations(
    *,
    model: SwinUNetDualViewSSL,
    x: torch.Tensor,
    plane_one_hot: torch.Tensor,
) -> Dict[str, Dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    out: Dict[str, Dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    pixel_mask = None
    x1 = model._apply_pixel_mask(x, pixel_mask)
    x2 = model._apply_pixel_mask(flip_lr(x), pixel_mask)

    f0_1 = model.patch_embed_1(x1)
    f0_2 = model.patch_embed_2(x2)
    out["after_patch_embed"] = {"pre": (f0_1, f0_2)}
    f0_1, f0_2 = model.maybe_saca("after_patch_embed", f0_1, f0_2)
    out["after_patch_embed"]["post"] = (f0_1, f0_2)

    s0_1 = model.stage0_1(f0_1)
    s0_2 = model.stage0_2(f0_2)
    out["after_stage0"] = {"pre": (s0_1, s0_2)}
    s0_1, s0_2 = model.maybe_saca("after_stage0", s0_1, s0_2)
    out["after_stage0"]["post"] = (s0_1, s0_2)

    f1_1 = model.merge0_1(s0_1)
    f1_2 = model.merge0_2(s0_2)
    out["after_merge0"] = {"pre": (f1_1, f1_2)}
    f1_1, f1_2 = model.maybe_saca("after_merge0", f1_1, f1_2)
    out["after_merge0"]["post"] = (f1_1, f1_2)

    s1_1 = model.stage1_1(f1_1)
    s1_2 = model.stage1_2(f1_2)
    out["after_stage1"] = {"pre": (s1_1, s1_2)}
    s1_1, s1_2 = model.maybe_saca("after_stage1", s1_1, s1_2)
    out["after_stage1"]["post"] = (s1_1, s1_2)

    # Keep forward path usage parity with training graph.
    if model.merge1 is not None and model.plane_cond is not None and model.stage2 is not None:
        u2_1 = model.merge1(s1_1)
        u2_1 = model.plane_cond(u2_1, plane_one_hot)
        model.stage2(u2_1)
    return out


def _flatten_tokens(feat: torch.Tensor) -> torch.Tensor:
    # NHWC [B,H,W,C] -> [H*W, C], B=1
    if feat.ndim != 4:
        raise ValueError(f"Expected NHWC tensor rank=4, got shape={tuple(feat.shape)}")
    return feat[0].reshape(-1, feat.shape[-1])


def _cosine_matrix(
    f1: torch.Tensor,
    f2: torch.Tensor,
    *,
    align_view2: bool,
) -> torch.Tensor:
    v1 = f1
    v2 = flip_lr_nhwc(f2) if align_view2 else f2
    a = _flatten_tokens(v1)
    b = _flatten_tokens(v2)
    a = F.normalize(a, dim=1, eps=1e-8)
    b = F.normalize(b, dim=1, eps=1e-8)
    return a @ b.transpose(0, 1)


def _feature_energy_map(feat: torch.Tensor) -> np.ndarray:
    # [1,H,W,C] -> [H,W] L2 norm over channels
    e = torch.linalg.norm(feat[0], dim=-1)
    arr = e.detach().cpu().numpy()
    arr = arr - float(arr.min())
    vmax = float(arr.max())
    if vmax > 1e-12:
        arr = arr / vmax
    return arr


def _best_match_map(cos: torch.Tensor, axis: int, hw: tuple[int, int]) -> np.ndarray:
    if axis == 0:
        v = cos.max(dim=1).values  # per patch in view1
    else:
        v = cos.max(dim=0).values  # per patch in view2
    arr = v.reshape(hw[0], hw[1]).detach().cpu().numpy()
    arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def _save_heatmap(arr: np.ndarray, out_path: Path, title: str, *, vmin: float = 0.0, vmax: float = 1.0) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    im = ax.imshow(arr, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Patch index (view2)")
    ax.set_ylabel("Patch index (view1)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_overlay(input_img: np.ndarray, heatmap: np.ndarray, out_path: Path, title: str) -> None:
    h, w = input_img.shape
    ten = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    ten = F.interpolate(ten, size=(h, w), mode="bilinear", align_corners=False)
    hm = ten.squeeze(0).squeeze(0).numpy()
    hm = hm - float(hm.min())
    vmax = float(hm.max())
    if vmax > 1e-12:
        hm = hm / vmax

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    ax.imshow(input_img, cmap="gray", vmin=0.0, vmax=1.0)
    ax.imshow(hm, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _choose_positions(
    *,
    model: SwinUNetDualViewSSL,
    target_position: str,
    plot_all_positions: bool,
) -> list[str]:
    if plot_all_positions:
        return list(SACA_POINTS)

    tp = str(target_position).strip().lower()
    if tp != "auto":
        if tp not in SACA_POINTS:
            raise ValueError(f"target_position must be one of {SACA_POINTS} or auto")
        return [tp]

    if model.enable_saca and model.saca_positions:
        ordered_enabled = [p for p in SACA_POINTS if p in set(model.saca_positions)]
        if ordered_enabled:
            return [ordered_enabled[-1]]
    return ["after_stage1"]


def _mean_np(arrs: list[np.ndarray]) -> np.ndarray:
    if not arrs:
        raise ValueError("Cannot aggregate empty array list.")

    target_h = max(int(a.shape[0]) for a in arrs)
    target_w = max(int(a.shape[1]) for a in arrs)
    resized: list[np.ndarray] = []
    for a in arrs:
        if int(a.shape[0]) == target_h and int(a.shape[1]) == target_w:
            resized.append(a)
            continue
        ten = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
        ten = F.interpolate(ten, size=(target_h, target_w), mode="bilinear", align_corners=False)
        resized.append(ten.squeeze(0).squeeze(0).cpu().numpy())

    x = np.stack(resized, axis=0).mean(axis=0)
    x = x - float(x.min())
    vmax = float(x.max())
    if vmax > 1e-12:
        x = x / vmax
    return x


def run(args: argparse.Namespace) -> None:
    cli_cfg = ExperimentConfig.from_args(args)
    if not str(cli_cfg.training.resume_ckpt).strip():
        raise ValueError("Missing --resume-ckpt.")

    ckpt_path = Path(cli_cfg.training.resume_ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    image_path = Path(args.input_image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model_cfg_source = str(args.model_config_source).strip().lower()
    cfg = cli_cfg if model_cfg_source == "cli" else _cfg_from_ckpt(ckpt_path)
    if int(args.plot_image_size) > 0:
        cfg.data.image_size = int(args.plot_image_size)
        cfg.mask.image_size = int(args.plot_image_size)

    if str(args.plot_out_dir).strip():
        out_dir = Path(args.plot_out_dir).expanduser().resolve()
    else:
        out_dir = Path(cli_cfg.logging.out_dir)
        if cli_cfg.logging.run_name:
            out_dir = out_dir / cli_cfg.logging.run_name
        out_dir = out_dir / "patch_cosine_similarity"
        out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(bool(args.cpu))
    model = _build_model(cfg, device)
    _load_weights(
        model=model,
        ckpt_path=ckpt_path,
        ckpt_load_mode=cli_cfg.training.ckpt_load_mode,
        device=device,
    )

    # Force analysis mode.
    model.enable_contrastive = False
    model.single_view = False
    if bool(args.activate_saca):
        model.current_epoch = max(int(getattr(model, "saca_warmup_epochs", 0)), 1)
    model.eval()

    x = _load_gray_tensor(image_path=image_path, image_size=int(cfg.data.image_size), device=device)
    plane_name = _parse_plane(args.plane, image_path, cfg.data.plane)
    plane = plane_to_one_hot(plane_name).to(device=device, dtype=torch.float32).view(1, 2)

    with torch.no_grad():
        reps = _extract_position_representations(model=model, x=x, plane_one_hot=plane)

    positions = _choose_positions(
        model=model,
        target_position=args.target_position,
        plot_all_positions=bool(args.plot_all_positions),
    )

    input_img = x[0, 0].detach().cpu().numpy()
    metadata: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "input_image": str(image_path),
        "plane": plane_name,
        "model_config_source": model_cfg_source,
        "ckpt_load_mode": str(cli_cfg.training.ckpt_load_mode),
        "enable_saca": bool(model.enable_saca),
        "saca_positions_cfg": list(getattr(model, "saca_positions", [])),
        "positions_plotted": list(positions),
        "records": [],
    }

    agg_pre = []
    agg_post = []
    agg_delta = []
    for pos in positions:
        if pos not in reps:
            continue
        pos_dir = out_dir / pos
        pos_dir.mkdir(parents=True, exist_ok=True)
        pre1, pre2 = reps[pos]["pre"]
        post1, post2 = reps[pos]["post"]

        pre_cos = _cosine_matrix(pre1, pre2, align_view2=bool(args.align_view2)).detach().cpu().numpy()
        post_cos = _cosine_matrix(post1, post2, align_view2=bool(args.align_view2)).detach().cpu().numpy()
        delta = post_cos - pre_cos

        # Cos matrix range in [-1,1].
        _save_heatmap(pre_cos, pos_dir / "cosine_pre.png", f"{pos} | cosine pre", vmin=-1.0, vmax=1.0)
        _save_heatmap(post_cos, pos_dir / "cosine_post.png", f"{pos} | cosine post", vmin=-1.0, vmax=1.0)
        _save_heatmap(delta, pos_dir / "cosine_delta.png", f"{pos} | cosine delta(post-pre)", vmin=-1.0, vmax=1.0)

        h, w, _ = pre1.shape[1:]
        pre_best = _best_match_map(torch.from_numpy(pre_cos), axis=0, hw=(h, w))
        post_best = _best_match_map(torch.from_numpy(post_cos), axis=0, hw=(h, w))
        e_pre = _feature_energy_map(pre1)
        e_post = _feature_energy_map(post1)
        _save_overlay(input_img, pre_best, pos_dir / "overlay_pre_best_match.png", f"{pos} | pre best-match")
        _save_overlay(input_img, post_best, pos_dir / "overlay_post_best_match.png", f"{pos} | post best-match")
        _save_overlay(input_img, e_pre, pos_dir / "overlay_pre_energy.png", f"{pos} | pre feature energy")
        _save_overlay(input_img, e_post, pos_dir / "overlay_post_energy.png", f"{pos} | post feature energy")

        if bool(args.save_npy):
            np.save(pos_dir / "cosine_pre.npy", pre_cos)
            np.save(pos_dir / "cosine_post.npy", post_cos)
            np.save(pos_dir / "cosine_delta.npy", delta)

        agg_pre.append((pre_cos + 1.0) / 2.0)
        agg_post.append((post_cos + 1.0) / 2.0)
        agg_delta.append(np.abs(delta))

        metadata["records"].append(
            {
                "position": pos,
                "shape": [int(pre1.shape[1]), int(pre1.shape[2]), int(pre1.shape[3])],
                "cos_pre": str(pos_dir / "cosine_pre.png"),
                "cos_post": str(pos_dir / "cosine_post.png"),
                "cos_delta": str(pos_dir / "cosine_delta.png"),
            }
        )

    if agg_pre:
        pre_m = _mean_np(agg_pre)
        post_m = _mean_np(agg_post)
        delta_m = _mean_np(agg_delta)
        _save_heatmap(pre_m, out_dir / "aggregate_cosine_pre.png", "aggregate cosine pre", vmin=0.0, vmax=1.0)
        _save_heatmap(post_m, out_dir / "aggregate_cosine_post.png", "aggregate cosine post", vmin=0.0, vmax=1.0)
        _save_heatmap(delta_m, out_dir / "aggregate_abs_delta.png", "aggregate abs delta", vmin=0.0, vmax=1.0)

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] saved patch cosine similarity outputs to: {out_dir}")


def _set_not_required(parser: argparse.ArgumentParser, dest: str) -> None:
    for action in parser._actions:
        if action.dest == dest:
            action.required = False


def build_argparser() -> argparse.ArgumentParser:
    p = build_ssl_argparser()
    p.description = "Patch-wise cosine similarity at SACA points (pre/post)"
    _set_not_required(p, "data_root")

    g = p.add_argument_group("patch_cosine")
    g.add_argument("--input-image", type=str, required=True, help="Single grayscale image path")
    g.add_argument("--plot-out-dir", type=str, default="", help="Output dir override")
    g.add_argument(
        "--model-config-source",
        type=str,
        default="ckpt",
        choices=["ckpt", "cli"],
        help="ckpt: build model from checkpoint cfg (recommended). cli: use current CLI args.",
    )
    g.add_argument(
        "--target-position",
        type=str,
        default="auto",
        choices=["auto", *SACA_POINTS],
        help="SACA point to visualize when --plot-all-positions is off.",
    )
    g.add_argument("--plot-all-positions", action="store_true", help="Plot all 4 SACA points.")
    g.add_argument("--align-view2", action="store_true", help="Flip view2 to align coordinates before cosine.")
    g.add_argument("--no-align-view2", dest="align_view2", action="store_false")
    g.set_defaults(align_view2=True)
    g.add_argument("--activate-saca", action="store_true", help="Set current_epoch >= warmup to force SACA active.")
    g.add_argument("--no-activate-saca", dest="activate_saca", action="store_false")
    g.set_defaults(activate_saca=True)
    g.add_argument("--plot-image-size", type=int, default=0, help="Optional image-size override for plotting.")
    g.add_argument("--save-npy", action="store_true", help="Also save cosine matrices as .npy")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
