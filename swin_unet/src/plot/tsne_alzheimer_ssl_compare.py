from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union, get_args, get_origin, get_type_hints

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402

from swin_unet.src.ver3.config.experiment import (
    ExperimentConfig,
    build_argparser as build_ssl_argparser,
)
from swin_unet.src.ver3.models.swin_unet_dualview_ssl import SwinUNetDualViewSSL
from swin_unet.src.ver3.training.ckpt_io import (
    load_checkpoint_weights,
    load_checkpoint_weights_filtered,
)
from swin_unet.src.ver3.training.utils import get_device


ENCODED_ID_TO_NAME: Dict[int, str] = {
    2: "Non_Demented",
    3: "Very_Mild_Demented",
    0: "Mild_Demented",
    1: "Moderate_Demented",
}

# Light -> heavy
SEVERITY_ORDER: Tuple[str, ...] = (
    "Non_Demented",
    "Very_Mild_Demented",
    "Mild_Demented",
    "Moderate_Demented",
)


def _normalize_label_name(name: str) -> str:
    s = str(name).lower().strip().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    if "moderate" in s:
        return "Moderate_Demented"
    if "very mild" in s:
        return "Very_Mild_Demented"
    if "mild" in s:
        return "Mild_Demented"
    if "non" in s:
        return "Non_Demented"
    return str(name).replace(" ", "_")


def _normalize_hex_color(color: str) -> str:
    c = str(color).strip()
    if not c.startswith("#"):
        c = f"#{c}"
    if len(c) != 7:
        raise ValueError(f"Invalid hex color: {color}. Expected #RRGGBB or RRGGBB.")
    return c


def _parse_color_map(spec: str | None, fallback: Dict[str, str]) -> Dict[str, str]:
    if not spec:
        return dict(fallback)
    out = dict(fallback)
    parts = [x.strip() for x in spec.split(",") if x.strip()]
    for p in parts:
        if ":" not in p:
            raise ValueError(
                f"Invalid color map segment '{p}'. Expected format 'Class Name:#RRGGBB'."
            )
        k, v = p.split(":", 1)
        out[_normalize_label_name(k.strip())] = _normalize_hex_color(v.strip())
    return out


class HFAlzheimerTestDataset(Dataset):
    def __init__(self, image_size: int):
        self.ds = load_dataset("Falah/Alzheimer_MRI", split="test")
        if "label" not in self.ds.features:
            raise RuntimeError("HuggingFace dataset is missing 'label' column.")
        self.label_names = list(self.ds.features["label"].names)
        self.tfm = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((int(image_size), int(image_size))),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.ds[int(idx)]
        x = self.tfm(item["image"])
        y = int(item["label"])
        return x, y


def _cfg_from_ckpt(ckpt_path: Path) -> ExperimentConfig:
    obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid checkpoint format at {ckpt_path}.")
    raw_cfg = obj.get("cfg", None)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Checkpoint {ckpt_path} has no 'cfg' dictionary.")
    return _dataclass_from_dict(ExperimentConfig, raw_cfg)


def _dataclass_from_dict(dc_type, raw: dict):
    """
    Rebuild nested dataclass from dict while tolerating checkpoint schema drift.
    """
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


def _build_model_from_cfg(cfg: ExperimentConfig, device: torch.device) -> SwinUNetDualViewSSL:
    mcfg = cfg.model
    dcfg = cfg.data

    model = SwinUNetDualViewSSL(
        in_ch=int(mcfg.in_ch),
        image_size=int(dcfg.image_size),
        patch_size=int(mcfg.patch_size),
        embed_dim=int(mcfg.embed_dim),
        enc_depths=tuple(mcfg.enc_depths),
        dec_depths=tuple(mcfg.dec_depths),
        num_heads=tuple(mcfg.num_heads),
        window_size=int(mcfg.window_size),
        proj_dim=int(mcfg.proj_dim),
        plane_inject_method=str(mcfg.plane_inject_method),
        enable_saca=bool(mcfg.enable_saca),
        saca_position=str(mcfg.saca_position),
        saca_positions=list(mcfg.saca_positions),
        saca_gate_init=float(mcfg.saca_gate_init),
        saca_warmup_epochs=int(mcfg.saca_warmup_epochs),
        enable_reconstruct=False,
        enable_contrastive=False,
        single_view=False,
    ).to(device)
    model.eval()
    return model


def _build_model_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
    *,
    cli_cfg: ExperimentConfig | None,
    image_size_override: int,
) -> SwinUNetDualViewSSL:
    cfg = cli_cfg if cli_cfg is not None else _cfg_from_ckpt(ckpt_path)

    if int(image_size_override) > 0:
        cfg.data.image_size = int(image_size_override)
        cfg.mask.image_size = int(image_size_override)

    model = _build_model_from_cfg(cfg, device=device)
    ckpt_mode = str(getattr(cfg.training, "ckpt_load_mode", "encoder_only"))
    _load_model_weights_with_mode(
        ckpt_path=ckpt_path,
        model=model,
        device=device,
        ckpt_mode=ckpt_mode,
    )
    return model


def _load_model_weights_with_mode(
    *,
    ckpt_path: Path,
    model: SwinUNetDualViewSSL,
    device: torch.device,
    ckpt_mode: str,
) -> None:
    mode = str(ckpt_mode or "encoder_only").lower().strip()
    if mode == "none":
        # Plotting requires loading weights; default to encoder_only when mode=none.
        print("[ckpt] ckpt_load_mode=none in cfg -> fallback to encoder_only for plotting.")
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
        raise ValueError(f"Unsupported ckpt_load_mode for plotting: {ckpt_mode}")

    # Use project-native encoder_only loading first.
    try:
        obj = load_checkpoint_weights_filtered(
            ckpt_path=ckpt_path,
            device=device,
            model=model,
            include_prefixes=model.encoder_state_dict_prefixes(),
            exclude_prefixes=("proj_c1", "proj_c2", "proj_c3", "proj"),
        )
        load_msg = obj.get("_load_msg", {}) if isinstance(obj, dict) else {}
        missing_n = len(load_msg.get("missing_keys", []))
        unexpected_n = len(load_msg.get("unexpected_keys", []))
        print(
            f"[ckpt] {ckpt_path.name}: encoder_only load done | "
            f"missing={missing_n} unexpected={unexpected_n}"
        )
        return
    except RuntimeError as e:
        print(f"[ckpt] encoder_only native load failed, fallback compatibility loader. reason={e}")
        _load_encoder_state_filtered_compat(ckpt_path=ckpt_path, model=model, device=device)


def _load_encoder_state_filtered_compat(
    *,
    ckpt_path: Path,
    model: SwinUNetDualViewSSL,
    device: torch.device,
) -> None:
    """
    Compatibility path:
    - encoder-only filtering
    - adapt legacy SACA gate shapes (scalar -> per-channel vector)
    - drop shape-mismatch keys
    """
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

        # Backward compatibility: old checkpoints used scalar SACA gate, new model uses per-channel gate.
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
    if dropped_shape:
        preview = ", ".join([f"{k}:{s}->{t}" for k, s, t in dropped_shape[:5]])
        print(f"[ckpt] dropped shape-mismatch keys preview: {preview}")


@torch.no_grad()
def _extract_embeddings(
    model: SwinUNetDualViewSSL,
    loader: DataLoader,
    device: torch.device,
    max_items: int,
) -> Tuple[np.ndarray, np.ndarray]:
    all_emb = []
    all_lab = []
    seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        bsz = int(x.shape[0])
        plane = torch.tensor([0.0, 1.0], device=device).view(1, 2).repeat(bsz, 1)
        feat = model.encode_bottleneck(x, plane, view=1)
        emb = feat.mean(dim=(1, 2))

        all_emb.append(emb.detach().cpu().numpy())
        all_lab.append(y.numpy())
        seen += bsz
        if max_items > 0 and seen >= max_items:
            break

    if not all_emb:
        raise RuntimeError("No embeddings were extracted (empty dataset or max_items=0).")

    emb_np = np.concatenate(all_emb, axis=0)
    lab_np = np.concatenate(all_lab, axis=0)
    if max_items > 0 and emb_np.shape[0] > max_items:
        emb_np = emb_np[:max_items]
        lab_np = lab_np[:max_items]
    return emb_np, lab_np


def _scatter_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: Iterable[str],
    color_map: Dict[str, str],
    out_path: Path,
    title: str,
    alpha: float,
    no_axes: bool,
    point_size: float,
) -> None:
    names = list(label_names)
    fig, ax = plt.subplots(figsize=(8, 8))

    present = sorted(set(int(x) for x in labels.tolist()))
    severity_rank = {name: i for i, name in enumerate(SEVERITY_ORDER)}
    ordered_ids = sorted(
        present,
        key=lambda lid: severity_rank.get(
            _normalize_label_name(
                ENCODED_ID_TO_NAME.get(lid, names[lid] if 0 <= lid < len(names) else f"class_{lid}")
            ),
            999,
        ),
    )

    for lid in ordered_ids:
        mask = labels == lid
        class_name = _normalize_label_name(
            ENCODED_ID_TO_NAME.get(lid, names[lid] if 0 <= lid < len(names) else f"class_{lid}")
        )
        color = color_map.get(class_name, "#4f4f4f")
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=float(point_size),
            c=color,
            alpha=float(alpha),
            edgecolors="none",
            label=class_name,
        )

    if no_axes:
        # Keep tick marks, hide only axis label text/numbers.
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, labeltop=False, labelright=False)
    else:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(title)
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _run_tsne(emb: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        init="pca",
        learning_rate="auto",
        random_state=int(random_state),
    )
    return tsne.fit_transform(emb)


def _run_pca(emb: np.ndarray, random_state: int) -> np.ndarray:
    pca = PCA(n_components=2, random_state=int(random_state))
    return pca.fit_transform(emb)


def _save_info_file(
    info_path: Path,
    *,
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    tag: str,
    color_map: Dict[str, str],
    meta: Dict[str, object],
) -> None:
    info_path.parent.mkdir(parents=True, exist_ok=True)
    payload_meta = dict(meta)
    payload_meta["tag"] = str(tag)
    payload_meta["color_map"] = dict(color_map)
    np.savez_compressed(
        info_path,
        coords=np.asarray(coords, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        label_names=np.asarray(label_names, dtype=object),
        meta_json=np.asarray([json.dumps(payload_meta)], dtype=object),
    )


def _load_info_file(info_path: Path) -> Dict[str, object]:
    if not info_path.exists():
        raise FileNotFoundError(f"Info file not found: {info_path}")
    with np.load(info_path, allow_pickle=True) as data:
        coords = np.asarray(data["coords"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        label_names = [str(x) for x in data["label_names"].tolist()]
        meta_json = str(data["meta_json"].tolist()[0]) if "meta_json" in data else "{}"
    meta = json.loads(meta_json)
    return {
        "coords": coords,
        "labels": labels,
        "label_names": label_names,
        "meta": meta,
        "color_map": dict(meta.get("color_map", {})),
        "tag": str(meta.get("tag", info_path.stem)),
    }


def _plot_single_from_info(
    info: Dict[str, object],
    *,
    out_dir: Path,
    alpha: float,
    point_size: float,
) -> None:
    coords = np.asarray(info["coords"])
    labels = np.asarray(info["labels"])
    label_names = list(info["label_names"])
    color_map = dict(info.get("color_map", {}))
    tag = str(info.get("tag", "model"))

    _scatter_plot(
        coords=coords,
        labels=labels,
        label_names=label_names,
        color_map=color_map,
        out_path=out_dir / f"{tag}_tsne_square.svg",
        title=f"{tag} SSL Encoder t-SNE (Test)",
        alpha=float(alpha),
        no_axes=False,
        point_size=float(point_size),
    )
    _scatter_plot(
        coords=coords,
        labels=labels,
        label_names=label_names,
        color_map=color_map,
        out_path=out_dir / f"{tag}_tsne_square_no_axes.svg",
        title=f"{tag} SSL Encoder t-SNE (Test)",
        alpha=float(alpha),
        no_axes=True,
        point_size=float(point_size),
    )


def _plot_compare_from_infos(
    baseline_info_path: Path,
    our_info_path: Path,
    *,
    out_dir: Path,
    alpha: float,
    point_size: float,
) -> None:
    base = _load_info_file(baseline_info_path)
    our = _load_info_file(our_info_path)

    _plot_single_from_info(base, out_dir=out_dir, alpha=alpha, point_size=point_size)
    _plot_single_from_info(our, out_dir=out_dir, alpha=alpha, point_size=point_size)

    meta = {
        "baseline_info": str(baseline_info_path),
        "our_info": str(our_info_path),
        "baseline_tag": str(base.get("tag", "baseline")),
        "our_tag": str(our.get("tag", "our")),
        "alpha": float(alpha),
        "point_size": float(point_size),
    }
    (out_dir / "tsne_compare_from_info_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _set_not_required(parser: argparse.ArgumentParser, dest: str) -> None:
    for action in parser._actions:
        if action.dest == dest:
            action.required = False


def _has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return option in parser._option_string_actions


def build_argparser() -> argparse.ArgumentParser:
    p = build_ssl_argparser()
    _set_not_required(p, "data_root")
    # Keep tsne script defaults while inheriting full ver3 CLI.
    p.set_defaults(
        out_dir="swin_unet/outputs/alzheimer_tsne",
        image_size=256,
        batch_size=32,
        num_workers=2,
        seed=42,
        cpu=False,
    )
    if not _has_option(p, "--image-size"):
        p.add_argument("--image-size", type=int, default=256, help="Input image size.")

    g = p.add_argument_group("tsne_compare")
    g.add_argument("--mode", type=str, default="single", choices=["single", "compare"], help="single: run one model and export info+plots. compare: use two info files and render both.")

    g.add_argument("--ckpt", type=Path, default=None, help="Checkpoint path for single mode.")
    g.add_argument("--tag", type=str, default="model", help="Tag for single model outputs, e.g. baseline or our.")

    g.add_argument("--baseline-info", type=Path, default=None, help="Baseline info file (.npz) for compare mode.")
    g.add_argument("--our-info", type=Path, default=None, help="Our info file (.npz) for compare mode.")

    g.add_argument(
        "--model-config-source",
        type=str,
        default="ckpt",
        choices=["ckpt", "cli"],
        help="ckpt: build model from its checkpoint cfg. cli: build from current inherited ver3 CLI args.",
    )
    g.add_argument("--info-out", type=Path, default=None, help="Info output file (.npz) for single mode. Default: out-dir/<tag>_tsne_info.npz")

    g.add_argument("--max-items", type=int, default=0, help="Max test samples to use (0 = all).")
    g.add_argument("--perplexity", type=float, default=30.0)
    g.add_argument("--pca", action="store_true", help="Use PCA(2D) instead of t-SNE.")
    g.add_argument("--alpha", type=float, default=0.8, help="Scatter alpha (default 0.8 = 80%).")
    g.add_argument("--point-size", type=float, default=14.0)
    g.add_argument(
        "--colors",
        type=str,
        default="Non_Demented:#66c2a4,Very_Mild_Demented:#31a354,Mild_Demented:#2ca25f,Moderate_Demented:#006d2c",
        help="Comma list for single mode: 'Class:#RRGGBB,...'",
    )
    if not _has_option(p, "--cpu"):
        g.add_argument("--cpu", action="store_true", help="Force CPU.")
    return p


def _resolve_info_path(info_arg: Path | None, out_dir: Path, tag: str) -> Path:
    if info_arg is not None:
        return info_arg.expanduser().resolve()
    return (out_dir / f"{tag}_tsne_info.npz").resolve()


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(args.mode) == "compare":
        if args.baseline_info is None or args.our_info is None:
            raise ValueError("compare mode requires --baseline-info and --our-info")
        _plot_compare_from_infos(
            Path(args.baseline_info).expanduser().resolve(),
            Path(args.our_info).expanduser().resolve(),
            out_dir=out_dir,
            alpha=float(args.alpha),
            point_size=float(args.point_size),
        )
        print(f"[done] compare plots generated at: {out_dir}")
        return

    # single mode
    if args.ckpt is None:
        raise ValueError("single mode requires --ckpt")

    device = get_device(bool(args.cpu))
    print(f"[device] {device}")

    ckpt = Path(args.ckpt).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    ds = HFAlzheimerTestDataset(image_size=int(args.image_size))
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    cli_cfg = None
    if str(args.model_config_source) == "cli":
        cli_cfg = ExperimentConfig.from_args(args)
        cli_cfg.training.enable_reconstruct = False
        cli_cfg.training.enable_contrastive = False

    model = _build_model_from_ckpt(
        ckpt,
        device=device,
        cli_cfg=cli_cfg,
        image_size_override=int(args.image_size),
    )

    emb, y = _extract_embeddings(
        model,
        loader,
        device=device,
        max_items=int(args.max_items),
    )
    if bool(args.pca):
        coords = _run_pca(emb, random_state=int(args.seed))
    else:
        coords = _run_tsne(emb, perplexity=float(args.perplexity), random_state=int(args.seed))

    colors = _parse_color_map(args.colors, fallback={})
    info_path = _resolve_info_path(args.info_out, out_dir=out_dir, tag=str(args.tag))

    meta = {
        "ckpt": str(ckpt),
        "model_config_source": str(args.model_config_source),
        "num_samples": int(y.shape[0]),
        "labels": [str(_normalize_label_name(x)) for x in ds.label_names],
        "encoded_id_to_name": {str(k): v for k, v in ENCODED_ID_TO_NAME.items()},
        "severity_order": list(SEVERITY_ORDER),
        "projection": "pca" if bool(args.pca) else "tsne",
        "perplexity": float(args.perplexity),
        "seed": int(args.seed),
    }

    _save_info_file(
        info_path,
        coords=coords,
        labels=y,
        label_names=[str(x) for x in ds.label_names],
        tag=str(args.tag),
        color_map=colors,
        meta=meta,
    )

    info = _load_info_file(info_path)
    _plot_single_from_info(
        info,
        out_dir=out_dir,
        alpha=float(args.alpha),
        point_size=float(args.point_size),
    )

    print(f"[done] single model outputs at: {out_dir}")
    print(f"[done] info file: {info_path}")

    # Optional auto-merge if both info files are provided and exist.
    if args.baseline_info is not None and args.our_info is not None:
        b = Path(args.baseline_info).expanduser().resolve()
        o = Path(args.our_info).expanduser().resolve()
        if b.is_file() and o.is_file():
            _plot_compare_from_infos(
                b,
                o,
                out_dir=out_dir,
                alpha=float(args.alpha),
                point_size=float(args.point_size),
            )
            print(f"[done] compare plots generated at: {out_dir}")


if __name__ == "__main__":
    main()
