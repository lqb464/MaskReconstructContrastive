from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402

from swin_unet.src.ver3.models.swin_unet_dualview_ssl import SwinUNetDualViewSSL
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


def _build_model_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
    image_size: int,
) -> SwinUNetDualViewSSL:
    obj = torch.load(ckpt_path, map_location=device)
    if not isinstance(obj, dict) or "model" not in obj:
        raise ValueError(f"Invalid checkpoint format at {ckpt_path}. Expected dict with key 'model'.")
    raw_cfg = obj.get("cfg", {})

    model_cfg = raw_cfg.get("model", {}) if isinstance(raw_cfg, dict) else {}
    data_cfg = raw_cfg.get("data", {}) if isinstance(raw_cfg, dict) else {}

    ckpt_image_size = int(data_cfg.get("image_size", image_size)) if isinstance(data_cfg, dict) else int(image_size)
    if int(image_size) > 0:
        ckpt_image_size = int(image_size)

    model = SwinUNetDualViewSSL(
        in_ch=int(model_cfg.get("in_ch", 1)),
        image_size=ckpt_image_size,
        patch_size=int(model_cfg.get("patch_size", 16)),
        embed_dim=int(model_cfg.get("embed_dim", 96)),
        enc_depths=tuple(model_cfg.get("enc_depths", (2, 2, 6, 2))),
        dec_depths=tuple(model_cfg.get("dec_depths", (6, 2, 2))),
        num_heads=tuple(model_cfg.get("num_heads", (3, 6, 12, 24))),
        window_size=int(model_cfg.get("window_size", 7)),
        proj_dim=int(model_cfg.get("proj_dim", 128)),
        plane_inject_method=str(model_cfg.get("plane_inject_method", "film")),
        enable_saca=bool(model_cfg.get("enable_saca", False)),
        saca_position=str(model_cfg.get("saca_position", "after_stage1")),
        saca_positions=list(model_cfg.get("saca_positions", [])) if isinstance(model_cfg.get("saca_positions", []), list) else [],
        saca_gate_init=float(model_cfg.get("saca_gate_init", 0.0)),
        saca_warmup_epochs=int(model_cfg.get("saca_warmup_epochs", 0)),
        enable_reconstruct=False,
        enable_contrastive=False,
        single_view=False,
    ).to(device)

    encoder_prefixes = model.encoder_state_dict_prefixes()
    sd_all = obj["model"]
    if not isinstance(sd_all, dict):
        raise ValueError(f"Checkpoint 'model' field at {ckpt_path} must be a state_dict dict.")
    sd_enc = {k: v for k, v in sd_all.items() if str(k).startswith(encoder_prefixes)}
    msg = model.load_state_dict(sd_enc, strict=False)
    if len(msg.unexpected_keys) > 0:
        print(f"[ckpt] {ckpt_path.name}: unexpected_keys={len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 0:
        print(f"[ckpt] {ckpt_path.name}: missing_keys={len(msg.missing_keys)} (expected with encoder-only load)")

    model.eval()
    return model


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
            _normalize_label_name(ENCODED_ID_TO_NAME.get(lid, names[lid] if 0 <= lid < len(names) else f"class_{lid}")),
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
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
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


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Compare SSL encoder t-SNE on HF Alzheimer_MRI test split (baseline vs our model).")
    p.add_argument("--baseline-ckpt", type=Path, required=True, help="Path to baseline SSL checkpoint (.pt).")
    p.add_argument("--our-ckpt", type=Path, required=True, help="Path to our SSL checkpoint (.pt).")
    p.add_argument("--out-dir", type=Path, default=Path("swin_unet/outputs/alzheimer_tsne"), help="Output directory.")
    p.add_argument("--image-size", type=int, default=256, help="Input resize for HF images.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-items", type=int, default=0, help="Max test samples to use (0 = all).")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.8, help="Scatter alpha (default 0.8 = 80%).")
    p.add_argument("--point-size", type=float, default=14.0)
    p.add_argument(
        "--baseline-colors",
        type=str,
        default="Non_Demented:#bdbdbd,Very_Mild_Demented:#8a8a8a,Mild_Demented:#4f4f4f,Moderate_Demented:#111111",
        help="Comma list: 'Class:#RRGGBB,...'",
    )
    p.add_argument(
        "--our-colors",
        type=str,
        default="Non_Demented:#66c2a4,Very_Mild_Demented:#31a354,Mild_Demented:#2ca25f,Moderate_Demented:#006d2c",
        help="Comma list: 'Class:#RRGGBB,...'",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(bool(args.cpu))
    print(f"[device] {device}")

    baseline_ckpt = Path(args.baseline_ckpt).expanduser().resolve()
    our_ckpt = Path(args.our_ckpt).expanduser().resolve()
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_ckpt}")
    if not our_ckpt.exists():
        raise FileNotFoundError(f"Our checkpoint not found: {our_ckpt}")

    ds = HFAlzheimerTestDataset(image_size=int(args.image_size))
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    baseline_model = _build_model_from_ckpt(baseline_ckpt, device=device, image_size=int(args.image_size))
    our_model = _build_model_from_ckpt(our_ckpt, device=device, image_size=int(args.image_size))

    emb_base, y = _extract_embeddings(
        baseline_model,
        loader,
        device=device,
        max_items=int(args.max_items),
    )
    emb_our, y2 = _extract_embeddings(
        our_model,
        loader,
        device=device,
        max_items=int(args.max_items),
    )
    if y.shape[0] != y2.shape[0] or (y != y2).any():
        raise RuntimeError("Label mismatch between baseline and our embedding extraction.")

    coords_base = _run_tsne(emb_base, perplexity=float(args.perplexity), random_state=int(args.seed))
    coords_our = _run_tsne(emb_our, perplexity=float(args.perplexity), random_state=int(args.seed))

    baseline_colors = _parse_color_map(args.baseline_colors, fallback={})
    our_colors = _parse_color_map(args.our_colors, fallback={})

    _scatter_plot(
        coords=coords_base,
        labels=y,
        label_names=ds.label_names,
        color_map=baseline_colors,
        out_path=out_dir / "baseline_tsne_square.png",
        title="Baseline SSL Encoder t-SNE (Test)",
        alpha=float(args.alpha),
        no_axes=False,
        point_size=float(args.point_size),
    )
    _scatter_plot(
        coords=coords_base,
        labels=y,
        label_names=ds.label_names,
        color_map=baseline_colors,
        out_path=out_dir / "baseline_tsne_square_no_axes.png",
        title="Baseline SSL Encoder t-SNE (Test)",
        alpha=float(args.alpha),
        no_axes=True,
        point_size=float(args.point_size),
    )
    _scatter_plot(
        coords=coords_our,
        labels=y,
        label_names=ds.label_names,
        color_map=our_colors,
        out_path=out_dir / "our_tsne_square.png",
        title="Our SSL Encoder t-SNE (Test)",
        alpha=float(args.alpha),
        no_axes=False,
        point_size=float(args.point_size),
    )
    _scatter_plot(
        coords=coords_our,
        labels=y,
        label_names=ds.label_names,
        color_map=our_colors,
        out_path=out_dir / "our_tsne_square_no_axes.png",
        title="Our SSL Encoder t-SNE (Test)",
        alpha=float(args.alpha),
        no_axes=True,
        point_size=float(args.point_size),
    )

    meta = {
        "baseline_ckpt": str(baseline_ckpt),
        "our_ckpt": str(our_ckpt),
        "num_samples": int(y.shape[0]),
        "labels": [str(_normalize_label_name(x)) for x in ds.label_names],
        "baseline_colors": baseline_colors,
        "our_colors": our_colors,
        "alpha": float(args.alpha),
        "perplexity": float(args.perplexity),
        "seed": int(args.seed),
    }
    (out_dir / "tsne_compare_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[done] wrote plots to: {out_dir}")
    print("[done] files:")
    print(f"  - {out_dir / 'baseline_tsne_square.png'}")
    print(f"  - {out_dir / 'baseline_tsne_square_no_axes.png'}")
    print(f"  - {out_dir / 'our_tsne_square.png'}")
    print(f"  - {out_dir / 'our_tsne_square_no_axes.png'}")


if __name__ == "__main__":
    main()
