import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

gt_dir = '../image_pairs/original'
pred_dir = '../image_pairs/your_result'

psnr_list = []

for i in range(1, 41):
    filename = f'{i}.png'
    gt_path = os.path.join(gt_dir, filename)
    pred_path = os.path.join(pred_dir, filename)

    gt_img = cv2.imread(gt_path)
    pred_img = cv2.imread(pred_path)

    if gt_img is None or pred_img is None:
        print(f"Missing file: {filename}")
        continue

    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

    if gt_img.shape != pred_img.shape:
        print(f"Size mismatch: {filename}")
        continue

    psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=255)

    print(f"{filename}: PSNR={psnr:.4f}")

    psnr_list.append(psnr)

if psnr_list:
    avg_psnr = np.mean(psnr_list)
    print(f"\nAverage PSNR: {avg_psnr:.4f}")
else:
    print("No valid images evaluated.")
