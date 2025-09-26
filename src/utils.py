import os
import gc
import numpy as np
import pandas as pd
import SimpleITK as sitk

def resample_image(image, new_spacing = (1.0,1.0,1.0), is_label = False):
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [int(round(orig_size[i] * (orig_spacing[i] / new_spacing[i]))) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    return resampler.Execute(image)

def create_mask_from_annotations(image, annotations_file, seriesuid):
    mask_np = np.zeros(image.GetSize()[::-1], dtype = np.uint8)
    df = pd.read_csv(annotations_file)
    df_ct = df[df['seriesuid'] == seriesuid]
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    for _, row in df_ct.iterrows():
        x_phys, y_phys, z_phys = row['coordX'], row['coordY'], row['coordZ']
        radius_mm = row['diameter_mm'] / 2
        x_idx = int(round((x_phys - origin[0]) / spacing[0]))
        y_idx = int(round((y_phys - origin[1]) / spacing[1]))
        z_idx = int(round((z_phys - origin[2]) / spacing[2]))
        r_x = int(round(radius_mm / spacing[0]))
        r_y = int(round(radius_mm / spacing[1]))
        r_z = int(round(radius_mm / spacing[2]))
        z_min, z_max = max(0, z_idx - r_z), min(mask_np.shape[0], z_idx + r_z + 1)
        y_min, y_max = max(0, y_idx - r_y), min(mask_np.shape[1], y_idx + r_y + 1)
        x_min, x_max = max(0, x_idx - r_x), min(mask_np.shape[2], x_idx + r_x + 1)
        zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
        dist2 = ((xx - x_idx) * spacing[0]) ** 2 + ((yy - y_idx) * spacing[1]) ** 2 + ((zz - z_idx) * spacing[2]) ** 2
        mask_np[z_min:z_max, y_min:y_max, x_min:x_max][dist2 <= radius_mm ** 2] = 1
    mask = sitk.GetImageFromArray(mask_np)
    mask.CopyInformation(image)
    return mask

def preprocess_and_save(mhd_folder, annotations_file, npy_folder, new_spacing = (1.0,1.0,1.0), max_files = None):
    files = sorted([f for f in os.listdir(mhd_folder) if f.endswith('.mhd')])
    if max_files:
        files = files[:max_files]
    for f in files:
        npy_img = os.path.join(npy_folder, f.replace(".mhd", "_img.npy"))
        npy_mask = os.path.join(npy_folder, f.replace(".mhd", "_mask.npy"))
        if os.path.exists(npy_img) and os.path.exists(npy_mask):
            continue
        img = sitk.ReadImage(os.path.join(mhd_folder, f))
        mask = create_mask_from_annotations(img, annotations_file, f.replace(".mhd",""))
        img_r = resample_image(img, new_spacing, is_label = False)
        mask_r = resample_image(mask, new_spacing, is_label = True)
        img_np = sitk.GetArrayFromImage(img_r).astype(np.float32)
        img_np = np.clip(img_np, -1000, 400)
        img_np = (img_np + 1000) / 1400
        mask_np = sitk.GetArrayFromImage(mask_r).astype(np.uint8)
        np.save(npy_img, img_np)
        np.save(npy_mask, mask_np)
        print(f"Saved {f} -> npy")
        del img, mask, img_r, mask_r, img_np, mask_np
        gc.collect()
