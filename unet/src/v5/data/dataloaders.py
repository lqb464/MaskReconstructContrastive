from __future__ import annotations

from typing import Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from .labels import build_label_config
from .datasets_hf import AlzheimerUNetDataset
from .datasets_adni_nifti import AdniNiftiSliceDataset
from .datasets_adni_precomputed import AdniPrecomputedSliceDataset
from .datasets_folder_csv import FolderUNetDataset


def create_unet_dataloader_from_folder_csv(
    csv_path: str,
    batch_size: int,
    image_size: int,
    num_workers: int,
    shuffle: bool = False,
    apply_unsharp: bool = False,
    unsharp_kernel_size: int = 5,
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 1.0,
    validate_images: bool = False,
) -> DataLoader:
    label_cfg = build_label_config()
    ds = FolderUNetDataset(
        csv_path=csv_path,
        image_size=image_size,
        label_cfg=label_cfg,
        validate_images=validate_images,
        apply_unsharp=apply_unsharp,
        unsharp_kernel_size=unsharp_kernel_size,
        unsharp_sigma=unsharp_sigma,
        unsharp_amount=unsharp_amount,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader



def create_unet_dataloaders(
    data_source: str,
    batch_size: int,
    image_size: int,
    num_workers: int,
    seed: int = 42,
    apply_unsharp: bool = False,
    pin_memory: bool = True,
    unsharp_kernel_size: int = 5,
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 1.0,
    adni_root_dir: Optional[str] = None,
    adni_image_type: str = "axial",
    adni_series_filter: Optional[str] = None,
    adni_label_csv: Optional[str] = None,
    adni_middle_frac: float = 0.4,
    adni_middle_subsample: int = 1,
    adni_preproc_root_dir: Optional[str] = None,
    folder_csv_path: Optional[str] = None,
    val_frac: float = 0.05,
    val_size: Optional[int] = None,  
) -> LoaderBundle:

    g = torch.Generator().manual_seed(seed)

    if val_size is not None:
            val_frac = max(1, int(val_size)) / max(1, n_total)

    if data_source == "hf":
        from datasets import load_dataset

        ds_train = load_dataset("Falah/Alzheimer_MRI", split="train")
        ds_test = load_dataset("Falah/Alzheimer_MRI", split="test")

        labels = [int(x["label"]) for x in ds_train]
        train_idx, val_idx = train_test_split(
            list(range(len(ds_train))),
            test_size=val_frac,
            random_state=seed,
            stratify=labels,
        )

        train_ds = AlzheimerUNetDataset(
            ds_train.select(train_idx),
            image_size=image_size,
            apply_unsharp=apply_unsharp,
            unsharp_kernel_size=unsharp_kernel_size,
            unsharp_sigma=unsharp_sigma,
            unsharp_amount=unsharp_amount,
        )
        val_ds = AlzheimerUNetDataset(
            ds_train.select(val_idx),
            image_size=image_size,
            apply_unsharp=apply_unsharp,
            unsharp_kernel_size=unsharp_kernel_size,
            unsharp_sigma=unsharp_sigma,
            unsharp_amount=unsharp_amount,
        )
        test_ds = AlzheimerUNetDataset(
            ds_test,
            image_size=image_size,
            apply_unsharp=apply_unsharp,
            unsharp_kernel_size=unsharp_kernel_size,
            unsharp_sigma=unsharp_sigma,
            unsharp_amount=unsharp_amount,
        )

    elif data_source == "adni":
        if adni_root_dir is None:
            raise ValueError("adni_root_dir is required for data_source='adni'")

        full_ds = AdniNiftiSliceDataset(
            root_dir=adni_root_dir,
            image_size=image_size,
            adni_image_type=adni_image_type,
            adni_series_filter=adni_series_filter,
            adni_label_csv=adni_label_csv,
            middle_frac=adni_middle_frac,
            middle_subsample=adni_middle_subsample,
            apply_unsharp=apply_unsharp,
            unsharp_kernel_size=unsharp_kernel_size,
            unsharp_sigma=unsharp_sigma,
            unsharp_amount=unsharp_amount,
        )

        n_total = len(full_ds)
        n_val = max(1, int(n_total * val_frac))
        n_train = n_total - n_val
        train_ds, valtmp_ds = random_split(full_ds, [n_train, n_val], generator=g)

        # Split valtmp into val and test
        n_val2 = n_val // 2
        n_test2 = n_val - n_val2
        val_ds, test_ds = random_split(valtmp_ds, [n_val2, n_test2], generator=g)

    elif data_source == "adni_preproc":
        if adni_preproc_root_dir is None:
            raise ValueError("adni_preproc_root_dir is required for data_source='adni_preproc'")

        full_ds = AdniPrecomputedSliceDataset(
            root_dir=adni_preproc_root_dir,
            image_size=image_size,
            apply_unsharp=apply_unsharp,
            unsharp_kernel_size=unsharp_kernel_size,
            unsharp_sigma=unsharp_sigma,
            unsharp_amount=unsharp_amount,
        )

        n_total = len(full_ds)
        n_val = max(1, int(n_total * val_frac))
        n_train = n_total - n_val
        train_ds, valtmp_ds = random_split(full_ds, [n_train, n_val], generator=g)

        n_val2 = n_val // 2
        n_test2 = n_val - n_val2
        val_ds, test_ds = random_split(valtmp_ds, [n_val2, n_test2], generator=g)

    elif data_source == "folder_csv":
        if folder_csv_path is None:
            raise ValueError("folder_csv_path is required for data_source='folder_csv'")

        label_cfg = build_label_config()
        full_ds = FolderUNetDataset(
            csv_path=folder_csv_path,
            image_size=image_size,
            label_cfg=label_cfg,
            validate_images=False,
            apply_unsharp=apply_unsharp,
            unsharp_kernel_size=unsharp_kernel_size,
            unsharp_sigma=unsharp_sigma,
            unsharp_amount=unsharp_amount,
        )

        n_total = len(full_ds)
        n_val = max(1, int(n_total * val_frac))
        n_train = n_total - n_val
        train_ds, valtmp_ds = random_split(full_ds, [n_train, n_val], generator=g)

        n_val2 = n_val // 2
        n_test2 = n_val - n_val2
        val_ds, test_ds = random_split(valtmp_ds, [n_val2, n_test2], generator=g)

    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    def make_loader(ds, shuffle: bool, drop_last: bool = False, pin_memory=False):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    train_loader = make_loader(train_ds, shuffle=True,  drop_last=True,  pin_memory=pin_memory)
    val_loader   = make_loader(val_ds,   shuffle=False, drop_last=False, pin_memory=pin_memory)
    test_loader  = make_loader(test_ds,  shuffle=False, drop_last=False, pin_memory=pin_memory)
 
    return LoaderBundle(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
