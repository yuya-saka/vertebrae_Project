#!/usr/bin/env python3
"""
Generate CSV files for segmentation and classification tasks from YOLO annotation files.
Creates both combined and orientation-specific CSV files:

Combined datasets:
1. segmentation_dataset.csv - For segmentation tasks (all orientations)
2. classification_dataset.csv - For binary classification tasks (all orientations)
3. fracture_level_dataset.csv - For fracture severity classification (all orientations)

Orientation-specific datasets (Axial, Coron, Sagit):
4-6. segmentation_dataset_{orientation}.csv - For segmentation tasks per orientation
7-9. classification_dataset_{orientation}.csv - For classification tasks per orientation
10-12. fracture_level_dataset_{orientation}.csv - For fracture level tasks per orientation
"""

import os
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse filename to extract patient_id, vertebra_id, slice_id.
    Example: A1003_V30_062_000.png -> {patient_id: A1003, vertebra_id: V30, slice_id: 062}
    """
    match = re.match(r'([A-Z]+\d+)_([V]\d+)_(\d+)_(\d+)\.png', filename)
    if match:
        return {
            'patient_id': match.group(1),
            'vertebra_id': match.group(2),
            'slice_id': match.group(3),
            'version': match.group(4)
        }
    return None


def parse_folder_name(folder_name: str) -> Optional[Dict[str, str]]:
    """
    Parse folder name to extract patient_id, vertebra_id, orientation.
    Example: slice_imageAI1003_vert30SagitAVGProjectionIntensity0208
    -> {patient_id: AI1003, vertebra: vert30, orientation: Sagit}
    """
    match = re.match(r'slice_image(AI\d+)_(vert\d+)(Axial|Coron|Sagit)', folder_name)
    if match:
        return {
            'patient_id': match.group(1),
            'vertebra': match.group(2),
            'orientation': match.group(3)
        }
    return None


def parse_ans_line(line: str) -> Tuple[str, Optional[List[int]]]:
    """
    Parse a line from ans_list file.
    Returns: (image_path, annotation_values or None)
    Example:
    - "/path/to/image.png" -> ("/path/to/image.png", None)
    - "/path/to/image.png 170,128,174,129,0,1" -> ("/path/to/image.png", [170,128,174,129,0,1])
    """
    parts = line.strip().split()
    if len(parts) == 0:
        return "", None

    image_path = parts[0]

    if len(parts) == 2:
        # Has annotation
        try:
            annotation = [int(x) for x in parts[1].split(',')]
            return image_path, annotation
        except ValueError:
            return image_path, None

    return image_path, None


def find_corresponding_mask_path(image_path: str, dataset_root: str) -> Optional[str]:
    """
    Find corresponding mask path for a given image path.
    Converts: slice_image/.../A1003_V30_062_000.png
    To: slice_image_ans/.../ans1003_V30_062_000.png
    """
    # Extract filename
    filename = os.path.basename(image_path)

    # Parse filename to get components
    parsed = parse_filename(filename)
    if not parsed:
        return None

    # Extract folder information
    folder_match = re.search(r'slice_image(AI\d+)_(vert\d+)(Axial|Coron|Sagit)', image_path)
    if not folder_match:
        return None

    patient_id = folder_match.group(1)
    vertebra = folder_match.group(2)
    orientation = folder_match.group(3)

    # Construct mask folder name
    mask_folder = f"slice_image_ans_{patient_id}_{vertebra}{orientation}"

    # Construct mask filename (ans prefix + lowercase patient_id)
    mask_filename = f"ans{patient_id[2:].lower()}_{parsed['vertebra_id']}_{parsed['slice_id']}_{parsed['version']}.png"

    # Full mask path
    mask_path = os.path.join(dataset_root, "slice_image_ans", mask_folder, mask_filename)

    # Verify file exists
    if os.path.exists(mask_path):
        return mask_path

    return None


def find_corresponding_rect_path(image_path: str, dataset_root: str) -> Optional[str]:
    """
    Find corresponding rect path for a given image path.
    Converts: slice_image/.../A1003_V30_062_000.png
    To: slice_image_rect/.../rect1003_V30_062_000.png
    """
    # Extract filename
    filename = os.path.basename(image_path)

    # Parse filename to get components
    parsed = parse_filename(filename)
    if not parsed:
        return None

    # Extract folder information
    folder_match = re.search(r'slice_image(AI\d+)_(vert\d+)(Axial|Coron|Sagit)', image_path)
    if not folder_match:
        return None

    patient_id = folder_match.group(1)
    vertebra = folder_match.group(2)
    orientation = folder_match.group(3)

    # Construct rect folder name
    rect_folder = f"slice_image_rectr{patient_id}_{vertebra}{orientation}"

    # Construct rect filename (rect prefix + lowercase patient_id)
    rect_filename = f"rect{patient_id[2:].lower()}_{parsed['vertebra_id']}_{parsed['slice_id']}_{parsed['version']}.png"

    # Full rect path
    rect_path = os.path.join(dataset_root, "slice_image_rect", rect_folder, rect_filename)

    # Verify file exists
    if os.path.exists(rect_path):
        return rect_path

    return None


def update_image_path(old_path: str, dataset_root: str) -> str:
    """
    Update image path from old location to new location.
    Old: /mnt/.../data/output/slice_imageAI1003_vert30Sagit.../A1003_V30_062_000.png
    New: /mnt/.../data/dataset/slice_image/slice_imageAI1003_vert30Sagit.../A1003_V30_062_000.png
    """
    # Extract the folder and filename
    match = re.search(r'(slice_image[^/]+)/([^/]+\.png)', old_path)
    if match:
        folder = match.group(1)
        filename = match.group(2)
        new_path = os.path.join(dataset_root, "slice_image", folder, filename)
        return new_path
    return old_path


def generate_datasets(dataset_root: str, ans_list_dir: str, output_dir: str):
    """
    Generate CSV files for different tasks, both combined and orientation-specific.
    """
    # Storage for all data
    all_data = []

    # Process all ans_list files
    ans_list_files = sorted([f for f in os.listdir(ans_list_dir) if f.startswith('ans_li')])

    print(f"Processing {len(ans_list_files)} ans_list files...")

    for ans_file in ans_list_files:
        ans_file_path = os.path.join(ans_list_dir, ans_file)

        with open(ans_file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                # Parse the line
                image_path, annotation = parse_ans_line(line)

                if not image_path:
                    continue

                # Update image path to new location
                updated_image_path = update_image_path(image_path, dataset_root)

                # Extract metadata from path
                folder_match = re.search(r'slice_image(AI\d+)_(vert\d+)(Axial|Coron|Sagit)', updated_image_path)
                if not folder_match:
                    continue

                patient_id = folder_match.group(1)
                vertebra_id = folder_match.group(2)
                orientation = folder_match.group(3)

                # Find corresponding mask and rect paths
                mask_path = find_corresponding_mask_path(updated_image_path, dataset_root)
                rect_path = find_corresponding_rect_path(updated_image_path, dataset_root)

                # Determine if has fracture
                has_fracture = 1 if annotation is not None else 0
                fracture_level = annotation[-1] if annotation else 0

                # Store data
                data_entry = {
                    'image_path': updated_image_path,
                    'mask_path': mask_path if mask_path else '',
                    'rect_path': rect_path if rect_path else '',
                    'patient_id': patient_id,
                    'vertebra_id': vertebra_id,
                    'orientation': orientation,
                    'has_fracture': has_fracture,
                    'fracture_level': fracture_level,
                    'annotation': annotation
                }

                all_data.append(data_entry)

    print(f"Processed {len(all_data)} total images")

    # Define orientations
    orientations = ['Axial', 'Coron', 'Sagit']

    # Generate segmentation datasets (combined and per-orientation)
    print("\n=== Generating Segmentation Datasets ===")

    # Combined segmentation dataset
    segmentation_csv = os.path.join(output_dir, 'segmentation_dataset.csv')
    with open(segmentation_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'mask_path', 'patient_id', 'vertebra_id', 'orientation', 'has_fracture'])

        for entry in all_data:
            writer.writerow([
                entry['image_path'],
                entry['mask_path'],
                entry['patient_id'],
                entry['vertebra_id'],
                entry['orientation'],
                entry['has_fracture']
            ])

    print(f"Generated: {segmentation_csv}")

    # Per-orientation segmentation datasets
    for orientation in orientations:
        orientation_data = [d for d in all_data if d['orientation'] == orientation]
        segmentation_csv_orient = os.path.join(output_dir, f'segmentation_dataset_{orientation.lower()}.csv')

        with open(segmentation_csv_orient, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'mask_path', 'patient_id', 'vertebra_id', 'orientation', 'has_fracture'])

            for entry in orientation_data:
                writer.writerow([
                    entry['image_path'],
                    entry['mask_path'],
                    entry['patient_id'],
                    entry['vertebra_id'],
                    entry['orientation'],
                    entry['has_fracture']
                ])

        print(f"Generated: {segmentation_csv_orient} ({len(orientation_data)} images)")

    # Generate classification datasets (combined and per-orientation)
    print("\n=== Generating Classification Datasets ===")

    # Combined classification dataset
    classification_csv = os.path.join(output_dir, 'classification_dataset.csv')
    with open(classification_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label', 'patient_id', 'vertebra_id', 'orientation'])

        for entry in all_data:
            writer.writerow([
                entry['image_path'],
                entry['has_fracture'],
                entry['patient_id'],
                entry['vertebra_id'],
                entry['orientation']
            ])

    print(f"Generated: {classification_csv}")

    # Per-orientation classification datasets
    for orientation in orientations:
        orientation_data = [d for d in all_data if d['orientation'] == orientation]
        classification_csv_orient = os.path.join(output_dir, f'classification_dataset_{orientation.lower()}.csv')

        with open(classification_csv_orient, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'label', 'patient_id', 'vertebra_id', 'orientation'])

            for entry in orientation_data:
                writer.writerow([
                    entry['image_path'],
                    entry['has_fracture'],
                    entry['patient_id'],
                    entry['vertebra_id'],
                    entry['orientation']
                ])

        print(f"Generated: {classification_csv_orient} ({len(orientation_data)} images)")

    # Generate fracture_level datasets (combined and per-orientation)
    print("\n=== Generating Fracture Level Datasets ===")

    # Combined fracture_level dataset
    fracture_level_csv = os.path.join(output_dir, 'fracture_level_dataset.csv')
    with open(fracture_level_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'mask_path', 'fracture_level', 'patient_id', 'vertebra_id', 'orientation'])

        fracture_count = 0
        for entry in all_data:
            if entry['has_fracture'] == 1:
                writer.writerow([
                    entry['image_path'],
                    entry['mask_path'],
                    entry['fracture_level'],
                    entry['patient_id'],
                    entry['vertebra_id'],
                    entry['orientation']
                ])
                fracture_count += 1

    print(f"Generated: {fracture_level_csv} (with {fracture_count} fracture samples)")

    # Per-orientation fracture_level datasets
    for orientation in orientations:
        orientation_fracture_data = [d for d in all_data if d['orientation'] == orientation and d['has_fracture'] == 1]
        fracture_level_csv_orient = os.path.join(output_dir, f'fracture_level_dataset_{orientation.lower()}.csv')

        with open(fracture_level_csv_orient, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'mask_path', 'fracture_level', 'patient_id', 'vertebra_id', 'orientation'])

            for entry in orientation_fracture_data:
                writer.writerow([
                    entry['image_path'],
                    entry['mask_path'],
                    entry['fracture_level'],
                    entry['patient_id'],
                    entry['vertebra_id'],
                    entry['orientation']
                ])

        print(f"Generated: {fracture_level_csv_orient} ({len(orientation_fracture_data)} fracture samples)")

    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total images: {len(all_data)}")
    print(f"Images with fractures: {sum(1 for d in all_data if d['has_fracture'] == 1)}")
    print(f"Images without fractures: {sum(1 for d in all_data if d['has_fracture'] == 0)}")
    print(f"Images with mask paths: {sum(1 for d in all_data if d['mask_path'])}")

    # Fracture level distribution
    fracture_levels = defaultdict(int)
    for d in all_data:
        if d['has_fracture'] == 1:
            fracture_levels[d['fracture_level']] += 1

    print("\nFracture level distribution:")
    for level in sorted(fracture_levels.keys()):
        print(f"  Level {level}: {fracture_levels[level]}")

    # Orientation distribution
    orientation_counts = defaultdict(int)
    orientation_fracture_counts = defaultdict(int)
    for d in all_data:
        orientation_counts[d['orientation']] += 1
        if d['has_fracture'] == 1:
            orientation_fracture_counts[d['orientation']] += 1

    print("\nOrientation distribution:")
    for orientation in orientations:
        total = orientation_counts.get(orientation, 0)
        fracture = orientation_fracture_counts.get(orientation, 0)
        print(f"  {orientation}: {total} total ({fracture} with fractures)")


if __name__ == '__main__':
    # Paths
    project_root = Path(__file__).parent.parent
    dataset_root = project_root / 'data' / 'dataset'
    ans_list_dir = dataset_root / 'ans_list'
    output_dir = dataset_root / 'Path'

    # Create output directory if not exists
    output_dir.mkdir(exist_ok=True)

    # Generate datasets
    generate_datasets(str(dataset_root), str(ans_list_dir), str(output_dir))

    print("\n✓ All CSV files generated successfully!")
