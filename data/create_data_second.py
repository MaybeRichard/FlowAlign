import os
import glob
import json
import random
import nibabel as nib

root_dir = os.getenv('', './data')

organ_indices = {
    "bladder": {
        "top_region_index": [0, 0, 1, 0],
        "bottom_region_index": [0, 0, 0, 1]
    },
    "bone": {
        "top_region_index": [0, 0, 0, 1],
        "bottom_region_index": [0, 0, 0, 1]
    },
    "cerebral": {
        "top_region_index": [1, 0, 0, 0],
        "bottom_region_index": [1, 0, 0, 0]
    },
    "colon": {
        "top_region_index": [0, 0, 1, 0],
        "bottom_region_index": [0, 0, 0, 1]
    },
    "esophageal": {
        "top_region_index": [1, 0, 0, 0],
        "bottom_region_index": [0, 0, 1, 0]
    },
    "gall": {
        "top_region_index": [0, 0, 1, 0],
        "bottom_region_index": [0, 0, 1, 0]
    },
    "gallbladder": {
        "top_region_index": [0, 0, 1, 0],
        "bottom_region_index": [0, 0, 1, 0]
    },
    "kidney": {
        "top_region_index": [0, 0, 1, 0],
        "bottom_region_index": [0, 0, 1, 0]
    },
    "liver": {
        "top_region_index": [0, 0, 1, 0],
        "bottom_region_index": [0, 0, 1, 0]
    },
    "lung": {
        "top_region_index": [0, 1, 0, 0],
        "bottom_region_index": [0, 1, 0, 0]
    },
    "pancreas": {
        "top_region_index": [0, 0, 1, 0],
        "bottom_region_index": [0, 0, 1, 0]
    },
    "stomach": {
        "top_region_index": [0, 0, 1, 0],
        "bottom_region_index": [0, 0, 1, 0]
    }
}

default_indices = {
    "top_region_index": [0, 0, 1, 0],
    "bottom_region_index": [0, 0, 0, 1]
}

sim_dim = [128, 128, 128]
sim_datalist = {"training": []}

organ_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

organ_to_label = {organ: idx for idx, organ in enumerate(sorted(organ_folders))}

for organ, label in organ_to_label.items():
    print(f"{organ}: {label}")

for organ_folder in organ_folders:
    organ_path = os.path.join(root_dir, organ_folder)
    class_label = organ_to_label[organ_folder]
    
    organ_name_parts = organ_folder.split('_')
    if len(organ_name_parts) > 0:
        organ_name = organ_name_parts[0].lower()
        if organ_name == "esophagus":
            organ_name = "esophageal"
    else:
        organ_name = organ_folder.lower()
    
    organ_specific_indices = organ_indices.get(organ_name, default_indices)
    
    case_folders = [f for f in os.listdir(organ_path) if os.path.isdir(os.path.join(organ_path, f))]
    total_cases = len(case_folders)
    fold_1_count = int(total_cases * 0.8)
    
    random.shuffle(case_folders)
    
    for idx, case_folder in enumerate(case_folders):
        case_path = os.path.join(organ_path, case_folder)
        
        plain_image_file = os.path.join(case_path, "plain_ct_emb.nii.gz")
        plain_label_file = os.path.join(case_path, "plain_bbox_mask.nii.gz")
        enhanced_image_file = os.path.join(case_path, "enhanced_ct.nii.gz")
        enhanced_label_file = os.path.join(case_path, "enhanced_bbox_mask.nii.gz")
        
        if (os.path.exists(plain_image_file) and os.path.exists(plain_label_file) and
            os.path.exists(enhanced_image_file) and os.path.exists(enhanced_label_file)):
            plain_image_file_rel = os.path.relpath(plain_image_file, root_dir)
            plain_label_file_rel = os.path.relpath(plain_label_file, root_dir)
            enhanced_image_file_rel = os.path.relpath(enhanced_image_file, root_dir)
            enhanced_label_file_rel = os.path.relpath(enhanced_label_file, root_dir)
            
            entry = {
                "plain_image": plain_image_file,
                "plain_label": plain_label_file,
                "enhanced_image": enhanced_image_file,
                "enhanced_label": enhanced_label_file,
                "fold": 1 if idx < fold_1_count else 0,
                "dim": sim_dim,
                "spacing": [1, 1, 1],
                "top_region_index": organ_specific_indices["top_region_index"],
                "bottom_region_index": organ_specific_indices["bottom_region_index"],
                "class_label": class_label,
            }
            
            sim_datalist["training"].append(entry)

class_counts = {}
for entry in sim_datalist["training"]:
    class_label = entry["class_label"]
    class_counts[class_label] = class_counts.get(class_label, 0) + 1

for class_label, count in sorted(class_counts.items()):
    organ = [k for k, v in organ_to_label.items() if v == class_label][0]
    print(f"Class {class_label} ({organ}): {count} samples")

datalist_file = os.path.join("./datasets", "pancreas.json")
os.makedirs(os.path.dirname(datalist_file), exist_ok=True)
with open(datalist_file, "w") as f:
    json.dump(sim_datalist, f, indent=4)

print(f"Generated dataset list saved to {datalist_file}")