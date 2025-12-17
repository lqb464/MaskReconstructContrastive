# data/labels.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


mindset_idx_map_label_1 = {
    '0': "Normal",
    '1': "MTL",
    '2': "Other",
    '3': "WMH",
}

mindset_label_map_idx_1 = {
    'mtl_atrophy': 1,              # Mild_Demented MTL
    'mtl_atrophy,other_atrophy': 1,# Moderate_Demented MTL
    'mtl_atrophy,wmh': 1,          # Very_Mild_Demented MTL
    'normal': 0,                   # Non_Demented N (nhãn 0) 4 màu hoàn toàn khác nhau 
    'other_atrophy': 2,            # Mild_Demented O
    'wmh': 3,                      # Very_Mild_Demented WMH
    'wmh,other_atrophy': 3         # Moderate_Demented WMH 
    # alzemer | normal | other 
}

mindset_colors_1 = {
    "MTL": "#df4122",
    "Normal": "#f6d097",
    "Other": "#7bbfc8",
    "WMH": "#f19809",
}


mindset_label_map_idx_2 = {
    'mtl_atrophy': 1,              # Mild_Demented MTL
    'mtl_atrophy,other_atrophy': 1,# Moderate_Demented MTL
    'mtl_atrophy,wmh': 1,          # Very_Mild_Demented MTL
    'normal': 0,                   # Non_Demented N (nhãn 0) 4 màu hoàn toàn khác nhau 
    'other_atrophy': 2,            # Mild_Demented O
    'wmh': 1,                      # Very_Mild_Demented WMH
    'wmh,other_atrophy': 1 
}

mindset_idx_map_label_2 = {
    '0': "Normal",
    '1': "MTL",
    '2': "Other",
}

mindset_colors_2 = {
    "MTL": "#df4122",
    "Normal": "#f6d097",
    "Other": "#7bbfc8",
}
    

# Huggingface mapping and fixed colors for t SNE legends
hf_idx_map_label = {
    '0': "Mild_Demented",
    '1': "Moderate_Demented",
    '2': "Non_Demented",
    '3': "Very_Mild_Demented",
}

hf_demantia_colors = {
    "Moderate_Demented": "#a5352b",
    "Non_Demented": "#457eb7",
    "Mild_Demented": "#e18775",
    "Very_Mild_Demented": "#ffe9c6",
}


