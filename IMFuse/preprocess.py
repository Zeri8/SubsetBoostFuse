import os
from argparse import ArgumentParser

import medpy.io as medio
import numpy as np


join = os.path.join


def resolve_nii(case_dir, case_id, suffix):
    nii_gz = os.path.join(case_dir, case_id + suffix + ".nii.gz")
    nii = os.path.join(case_dir, case_id + suffix + ".nii")
    if os.path.exists(nii_gz):
        return nii_gz
    if os.path.exists(nii):
        return nii
    raise FileNotFoundError(f"Missing file for {case_id}{suffix}: {nii_gz} or {nii}")


def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        print("#" * 100)
        ecart = int((128 - (xmax - xmin)) / 2)
        xmax = xmax + ecart + 1
        xmin = xmin - ecart
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    return xmin, xmax


def crop(vol):
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3

    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max


def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        vol[k, ...] = x
    return vol


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    src_path = args.input_path
    tar_path = args.output_path
    vol_out = os.path.join(tar_path, "vol")
    seg_out = os.path.join(tar_path, "seg")
    os.makedirs(vol_out, exist_ok=True)
    os.makedirs(seg_out, exist_ok=True)

    failed_cases = []
    name_list = sorted(os.listdir(src_path))

    for file_name in name_list:
        try:
            case_dir = os.path.join(src_path, file_name)
            if not os.path.isdir(case_dir):
                continue

            case_id = file_name
            print(case_id)

            flair, flair_header = medio.load(resolve_nii(case_dir, case_id, "-t2f"))
            t1ce, t1ce_header = medio.load(resolve_nii(case_dir, case_id, "-t1c"))
            t1, t1_header = medio.load(resolve_nii(case_dir, case_id, "-t1n"))
            t2, t2_header = medio.load(resolve_nii(case_dir, case_id, "-t2w"))

            vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)
            x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
            vol1 = normalize(vol[:, x_min:x_max, y_min:y_max, z_min:z_max])
            vol1 = vol1.transpose(1, 2, 3, 0)
            print(vol1.shape)

            seg, seg_header = medio.load(resolve_nii(case_dir, case_id, "-seg"))
            seg = seg.astype(np.uint8)
            seg1 = seg[x_min:x_max, y_min:y_max, z_min:z_max]
            seg1[seg1 == 4] = 3

            np.save(os.path.join(vol_out, case_id + "_vol.npy"), vol1)
            np.save(os.path.join(seg_out, case_id + "_seg.npy"), seg1)
        except Exception as exc:
            print(f"[SKIP] {file_name}: {exc}")
            failed_cases.append((file_name, str(exc)))

    if failed_cases:
        failed_path = os.path.join(tar_path, "failed_cases.txt")
        with open(failed_path, "w") as f:
            for case_id, err in failed_cases:
                f.write(f"{case_id}\t{err}\n")
        print(f"Failed cases: {len(failed_cases)}")
        print(f"Saved failed case list to: {failed_path}")
