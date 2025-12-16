# data/labels.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class LabelConfig:
    hf_label_map: Dict[int, str]
    hf_label_colors: Dict[int, str]
    mindset_label_map_idx_1: Dict[str, int]
    mindset_label_map_idx_2: Dict[str, int]
    mindset_label_colors_1: Dict[int, str]
    mindset_label_colors_2: Dict[int, str]


def build_label_config() -> LabelConfig:
    # Bạn chỉnh mapping theo đúng dự án của bạn nếu khác
    hf_label_map = {
        0: "NonDemented",
        1: "VeryMildDemented",
        2: "MildDemented",
        3: "ModerateDemented",
    }
    hf_label_colors = {
        0: "tab:blue",
        1: "tab:orange",
        2: "tab:green",
        3: "tab:red",
    }

    mindset_label_map_idx_1 = {
        "Normal": 0,
        "MTL": 1,
        "Other": 2,
        "WMH": 3,
    }
    mindset_label_map_idx_2 = {
        "Normal": 0,
        "MTL": 1,
        "Other": 2,
        "WMH": 3,
    }

    # Nếu bạn thật sự có 2 bộ màu khác nhau thì sửa ở đây
    mindset_label_colors_1 = {
        0: "tab:blue",
        1: "tab:orange",
        2: "tab:green",
        3: "tab:red",
    }
    mindset_label_colors_2 = {
        0: "tab:blue",
        1: "tab:orange",
        2: "tab:green",
        3: "tab:red",
    }

    return LabelConfig(
        hf_label_map=hf_label_map,
        hf_label_colors=hf_label_colors,
        mindset_label_map_idx_1=mindset_label_map_idx_1,
        mindset_label_map_idx_2=mindset_label_map_idx_2,
        mindset_label_colors_1=mindset_label_colors_1,
        mindset_label_colors_2=mindset_label_colors_2,
    )
