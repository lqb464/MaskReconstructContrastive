from __future__ import annotations

import argparse
import json
import math
import types
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union, get_args, get_origin, get_type_hints

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from swin_unet.src.ver3.config.experiment import ExperimentConfig
from swin_unet.src.ver3.data.dataset import plane_to_one_hot
from swin_unet.src.ver3.models.swin_unet_dualview_ssl import (
    SwinUNetDualViewSSL,
    WindowAttention,
    WindowCrossAttention,
)
from swin_unet.src.ver3.training.ckpt_io import (
    load_checkpoint_weights,
    load_checkpoint_weights_filtered,
)
from swin_unet.src.ver3.training.utils import get_device


def _dataclass_from_dict(dc_type, raw: dict):
    if not is_dataclass(dc_type):
        raise TypeError(f"{dc_type} is not a dataclass")

    type_hints = get_type_hints(dc_type)
    kwargs = {}
    for f in fields(dc_type):
        if f.name not in raw:
            continue
        val = raw[f.name]
        ftype = type_hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _dataclass_from_dict(ftype, val)
            continue
        origin = get_origin(ftype)
        args = get_args(ftype)
        if origin is Union and isinstance(val, dict):
            dc_candidates = [a for a in args if is_dataclass(a)]
            if dc_candidates:
                kwargs[f.name] = _dataclass_from_dict(dc_candidates[0], val)
                continue
        kwargs[f.name] = val
    return dc_type(**kwargs)


def _cfg_from_checkpoint(ckpt_path: Path) -> ExperimentConfig:
    obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid checkpoint object in {ckpt_path}")
    raw_cfg = obj.get("cfg", None)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Checkpoint {ckpt_path} has no cfg dictionary")
    return _dataclass_from_dict(ExperimentConfig, raw_cfg)


def _build_model(cfg: ExperimentConfig, device: torch.device) -> SwinUNetDualViewSSL:
    model = SwinUNetDualViewSSL(
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
        saca_positions=cfg.model.saca_positions,
        saca_gate_init=cfg.model.saca_gate_init,
        saca_warmup_epochs=cfg.model.saca_warmup_epochs,
        enable_reconstruct=cfg.training.enable_reconstruct,
        enable_contrastive=cfg.training.enable_contrastive,
        contrastive_loss_type=cfg.contrast_loss.contrastive_loss_type,
        contrastive_position=cfg.contrast_loss.contrastive_position,
        single_view=cfg.training.single_view,
    ).to(device)
    return model


def _load_weights(
    *,
    model: SwinUNetDualViewSSL,
    ckpt_path: Path,
    ckpt_load_mode: str,
    device: torch.device,
) -> None:
    mode = str(ckpt_load_mode).strip().lower()
    if mode == "none":
        return
    if mode == "full":
        load_checkpoint_weights(
            ckpt_path=ckpt_path,
            device=device,
            model=model,
            strict=True,
        )
        return
    if mode == "encoder_only":
        load_checkpoint_weights_filtered(
            ckpt_path=ckpt_path,
            device=device,
            model=model,
            include_prefixes=model.encoder_state_dict_prefixes(),
            exclude_prefixes=("proj_c1", "proj_c2", "proj_c3", "proj"),
        )
        return
    raise ValueError(f"Unsupported ckpt-load-mode={ckpt_load_mode}")


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


class AttentionCapture:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.records: list[dict[str, Any]] = []
        self._orig: dict[str, Any] = {}
        self._name_by_module_id = {id(m): n for n, m in model.named_modules()}

    def _record(
        self,
        *,
        module: torch.nn.Module,
        attn_prob: torch.Tensor,
        kind: str,
    ) -> None:
        name = self._name_by_module_id.get(id(module), module.__class__.__name__)
        self.records.append(
            {
                "name": str(name),
                "kind": str(kind),
                "window_size": int(getattr(module, "window_size", 0)),
                "attn": attn_prob.detach().to(dtype=torch.float32).cpu(),
            }
        )

    def enable(self) -> None:
        for module in self.model.modules():
            if isinstance(module, WindowAttention):
                self._patch_window_attention(module)
            elif isinstance(module, WindowCrossAttention):
                self._patch_cross_attention(module)

    def disable(self) -> None:
        for module_id, orig in self._orig.items():
            module = orig["module"]
            module.forward = orig["forward"]  # type: ignore[method-assign]
        self._orig.clear()

    def _patch_window_attention(self, module: WindowAttention) -> None:
        module_id = str(id(module))
        if module_id in self._orig:
            return
        self._orig[module_id] = {"module": module, "forward": module.forward}

        def _forward_with_capture(this: WindowAttention, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            bn, n, c = x.shape
            qkv = this.qkv(x).reshape(bn, n, 3, this.num_heads, c // this.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = q * this.scale
            attn = q @ k.transpose(-2, -1)

            rel_bias = this.relative_position_bias_table[this.relative_position_index.view(-1)].view(n, n, -1)
            rel_bias = rel_bias.permute(2, 0, 1).contiguous()
            attn = attn + rel_bias.unsqueeze(0)

            if attn_mask is not None:
                if attn_mask.dtype != attn.dtype:
                    attn_mask = attn_mask.to(dtype=attn.dtype)
                n_w = attn_mask.size(0)
                attn = attn.view(bn // n_w, n_w, this.num_heads, n, n)
                attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, this.num_heads, n, n)

            attn_prob = attn.softmax(dim=-1)
            out = (this.attn_drop(attn_prob) @ v).transpose(1, 2).reshape(bn, n, c)
            out = this.proj_drop(this.proj(out))
            self._record(module=this, attn_prob=attn_prob, kind="swin")
            return out

        module.forward = types.MethodType(_forward_with_capture, module)  # type: ignore[method-assign]

    def _patch_cross_attention(self, module: WindowCrossAttention) -> None:
        module_id = str(id(module))
        if module_id in self._orig:
            return
        self._orig[module_id] = {"module": module, "forward": module.forward}

        def _forward_with_capture(this: WindowCrossAttention, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
            bn, n, c = x_q.shape
            q = this.q(x_q).reshape(bn, n, this.num_heads, c // this.num_heads).permute(0, 2, 1, 3)
            kv = this.kv(x_kv).reshape(bn, n, 2, this.num_heads, c // this.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            q = q * this.scale
            attn = q @ k.transpose(-2, -1)

            rel_bias = this.relative_position_bias_table[this.relative_position_index.view(-1)].view(n, n, -1)
            rel_bias = rel_bias.permute(2, 0, 1).contiguous()
            attn = attn + rel_bias.unsqueeze(0)

            attn_prob = attn.softmax(dim=-1)
            out = (this.attn_drop(attn_prob) @ v).transpose(1, 2).reshape(bn, n, c)
            out = this.proj_drop(this.proj(out))
            self._record(module=this, attn_prob=attn_prob, kind="saca")
            return out

        module.forward = types.MethodType(_forward_with_capture, module)  # type: ignore[method-assign]


def _window_grid_shape(num_windows: int) -> tuple[int, int]:
    side = int(round(math.sqrt(float(num_windows))))
    if side > 0 and side * side == num_windows:
        return side, side
    for h in range(side, 0, -1):
        if num_windows % h == 0:
            return h, num_windows // h
    return num_windows, 1


def _attention_to_map(attn: torch.Tensor, window_size: int) -> np.ndarray:
    # attn: [Bn, heads, N, N]
    key_importance = attn.mean(dim=1).mean(dim=2)  # [Bn, N]
    b_windows = int(key_importance.shape[0])
    n = int(key_importance.shape[1])
    ws = int(window_size)
    if ws <= 0 or ws * ws != n:
        ws = int(round(math.sqrt(float(n))))
    gh, gw = _window_grid_shape(b_windows)
    canvas = torch.zeros((gh * ws, gw * ws), dtype=key_importance.dtype)

    for idx in range(b_windows):
        r = idx // gw
        c = idx % gw
        patch = key_importance[idx].reshape(ws, ws)
        canvas[r * ws : (r + 1) * ws, c * ws : (c + 1) * ws] = patch

    arr = canvas.numpy()
    arr = arr - float(arr.min())
    vmax = float(arr.max())
    if vmax > 1e-12:
        arr = arr / vmax
    return arr


def _resize_map_to_image(map_arr: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    ten = torch.from_numpy(map_arr).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    ten = F.interpolate(ten, size=(h, w), mode="bilinear", align_corners=False)
    out = ten.squeeze(0).squeeze(0).numpy()
    out = out - float(out.min())
    vmax = float(out.max())
    if vmax > 1e-12:
        out = out / vmax
    return out


def _save_overlay(input_img: np.ndarray, heatmap: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    ax.imshow(input_img, cmap="gray", vmin=0.0, vmax=1.0)
    ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_heatmap(heatmap: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    im = ax.imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _mean_maps(maps: Iterable[np.ndarray]) -> np.ndarray:
    arr = np.stack(list(maps), axis=0)
    out = arr.mean(axis=0)
    out = out - float(out.min())
    vmax = float(out.max())
    if vmax > 1e-12:
        out = out / vmax
    return out


def _saca_position(module_name: str) -> str:
    tag = "saca_modules."
    if tag not in module_name:
        return "unknown"
    tail = module_name.split(tag, 1)[1]
    return tail.split(".", 1)[0]


def run(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.resume_ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    image_path = Path(args.input_image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_module").mkdir(parents=True, exist_ok=True)

    device = get_device(cpu=bool(args.cpu))
    cfg = _cfg_from_checkpoint(ckpt_path)
    if int(args.image_size) > 0:
        cfg.data.image_size = int(args.image_size)

    model = _build_model(cfg, device=device)
    _load_weights(
        model=model,
        ckpt_path=ckpt_path,
        ckpt_load_mode=args.ckpt_load_mode,
        device=device,
    )

    # Force reconstruction-only + dual-view for visualization run.
    model.enable_contrastive = False
    model.single_view = False
    model.eval()

    x = _load_gray_tensor(image_path=image_path, image_size=int(cfg.data.image_size), device=device)
    plane_name = _parse_plane(args.plane, image_path, cfg.data.plane)
    plane = plane_to_one_hot(plane_name).to(device=device, dtype=torch.float32).view(1, 2)

    capture = AttentionCapture(model)
    capture.enable()
    try:
        with torch.no_grad():
            model(x, pixel_mask=None, plane_one_hot=plane)
    finally:
        capture.disable()

    if not capture.records:
        raise RuntimeError("No attention records were captured.")

    input_img = x.squeeze(0).squeeze(0).detach().cpu().numpy()
    image_hw = (int(input_img.shape[0]), int(input_img.shape[1]))

    module_maps: list[dict[str, Any]] = []
    for rec in capture.records:
        mod_map = _attention_to_map(rec["attn"], rec["window_size"])
        mod_map = _resize_map_to_image(mod_map, image_hw=image_hw)
        module_maps.append(
            {
                "name": rec["name"],
                "kind": rec["kind"],
                "window_size": rec["window_size"],
                "map": mod_map,
            }
        )

    base_maps = [x["map"] for x in module_maps if x["kind"] == "swin"]
    saca_maps = [x for x in module_maps if x["kind"] == "saca"]

    if not base_maps:
        print("[warn] No base Swin attention maps captured.")
    if cfg.model.enable_saca and not saca_maps:
        print("[warn] SACA is enabled in config but no SACA maps were captured.")

    metadata: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "input_image": str(image_path),
        "plane": plane_name,
        "image_size": int(cfg.data.image_size),
        "ckpt_load_mode": str(args.ckpt_load_mode),
        "num_records": int(len(module_maps)),
        "records": [],
        "saca_positions": sorted(set(_saca_position(x["name"]) for x in saca_maps)),
    }

    for idx, rec in enumerate(module_maps):
        stem = f"{idx:03d}_{rec['name'].replace('.', '_')}"
        hm_path = out_dir / "per_module" / f"{stem}_heatmap.png"
        ov_path = out_dir / "per_module" / f"{stem}_overlay.png"
        _save_heatmap(rec["map"], hm_path, title=f"{rec['kind']} | {rec['name']}")
        _save_overlay(input_img, rec["map"], ov_path, title=f"{rec['kind']} | {rec['name']}")
        if bool(args.save_npy):
            np.save(out_dir / "per_module" / f"{stem}.npy", rec["map"])
        metadata["records"].append(
            {
                "name": rec["name"],
                "kind": rec["kind"],
                "window_size": int(rec["window_size"]),
                "heatmap": str(hm_path),
                "overlay": str(ov_path),
            }
        )

    if base_maps:
        base_agg = _mean_maps(base_maps)
        _save_heatmap(base_agg, out_dir / "base_attention_aggregate_heatmap.png", title="Base Swin Attention (Aggregate)")
        _save_overlay(input_img, base_agg, out_dir / "base_attention_aggregate_overlay.png", title="Base Swin Attention (Aggregate)")
        if bool(args.save_npy):
            np.save(out_dir / "base_attention_aggregate.npy", base_agg)

    by_pos: dict[str, list[np.ndarray]] = {}
    by_dir: dict[str, list[np.ndarray]] = {}
    for rec in saca_maps:
        pos = _saca_position(rec["name"])
        by_pos.setdefault(pos, []).append(rec["map"])
        if rec["name"].endswith("xattn_12"):
            by_dir.setdefault(f"{pos}:12", []).append(rec["map"])
        elif rec["name"].endswith("xattn_21"):
            by_dir.setdefault(f"{pos}:21", []).append(rec["map"])

    pos_agg_maps: list[np.ndarray] = []
    for pos, maps in sorted(by_pos.items()):
        pos_agg = _mean_maps(maps)
        pos_agg_maps.append(pos_agg)
        _save_heatmap(pos_agg, out_dir / f"saca_{pos}_aggregate_heatmap.png", title=f"SACA {pos} (Aggregate)")
        _save_overlay(input_img, pos_agg, out_dir / f"saca_{pos}_aggregate_overlay.png", title=f"SACA {pos} (Aggregate)")
        if bool(args.save_npy):
            np.save(out_dir / f"saca_{pos}_aggregate.npy", pos_agg)

        m12 = by_dir.get(f"{pos}:12", [])
        m21 = by_dir.get(f"{pos}:21", [])
        if m12:
            a12 = _mean_maps(m12)
            _save_heatmap(a12, out_dir / f"saca_{pos}_dir12_heatmap.png", title=f"SACA {pos} dir12")
            _save_overlay(input_img, a12, out_dir / f"saca_{pos}_dir12_overlay.png", title=f"SACA {pos} dir12")
        if m21:
            a21 = _mean_maps(m21)
            _save_heatmap(a21, out_dir / f"saca_{pos}_dir21_heatmap.png", title=f"SACA {pos} dir21")
            _save_overlay(input_img, a21, out_dir / f"saca_{pos}_dir21_overlay.png", title=f"SACA {pos} dir21")

    if pos_agg_maps:
        saca_all = _mean_maps(pos_agg_maps)
        _save_heatmap(saca_all, out_dir / "saca_all_positions_aggregate_heatmap.png", title="SACA All Positions (Aggregate)")
        _save_overlay(input_img, saca_all, out_dir / "saca_all_positions_aggregate_overlay.png", title="SACA All Positions (Aggregate)")
        if bool(args.save_npy):
            np.save(out_dir / "saca_all_positions_aggregate.npy", saca_all)

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] attention heatmaps saved to: {out_dir}")
    print(f"[done] records captured: {len(module_maps)}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Visualize Swin/SACA attention heatmaps from SSL checkpoint")
    p.add_argument("--resume-ckpt", type=str, required=True, help="Path to SSL checkpoint (.pt)")
    p.add_argument(
        "--ckpt-load-mode",
        type=str,
        default="full",
        choices=["none", "full", "encoder_only"],
        help="Checkpoint loading mode (same semantics as ver3 experiment.py)",
    )
    p.add_argument("--input-image", type=str, required=True, help="Path to single grayscale image")
    p.add_argument("--out-dir", type=str, default="swin_unet/outputs/attention_heatmap_ssl")
    p.add_argument("--plane", type=str, default="auto", choices=["axial", "coronal", "auto"])
    p.add_argument("--image-size", type=int, default=0, help="Override image size (0 uses checkpoint cfg)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-npy", action="store_true", help="Also save raw heatmap arrays as .npy")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
