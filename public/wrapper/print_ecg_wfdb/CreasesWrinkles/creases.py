import cv2
import numpy as np
import random
import os
from pathlib import Path
from skimage import util
from PIL import Image
import heapq
import math

# =========================================================
# Resolve resource directories (PACKAGE SAFE)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
WRINKLES_DIR = BASE_DIR / "wrinkles-dataset"

if not WRINKLES_DIR.is_dir():
    raise FileNotFoundError(
        f"[CreasesWrinkles] wrinkles-dataset not found at: {WRINKLES_DIR}"
    )

# =========================================================
# Utility functions (UNCHANGED LOGIC)
# =========================================================

def randomPatch(texture, block_size):
    h, w, _ = texture.shape
    i = random.randint(0, h - block_size)
    j = random.randint(0, w - block_size)
    return texture[i:i+block_size, j:j+block_size]

def L2OverlapDiff(patch, block_size, overlap, res, y, x):
    error = 0
    if x > 0:
        left = patch[:, :overlap] - res[y:y+block_size, x:x+overlap]
        error += np.sum(left**2)
    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+block_size]
        error += np.sum(up**2)
    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y+overlap, x:x+overlap]
        error -= np.sum(corner**2)
    return error

def randomBestPatch(texture, block_size, overlap, res, y, x):
    h, w, _ = texture.shape
    errors = np.zeros((h - block_size, w - block_size))
    for i in range(h - block_size):
        for j in range(w - block_size):
            patch = texture[i:i+block_size, j:j+block_size]
            errors[i, j] = L2OverlapDiff(patch, block_size, overlap, res, y, x)
    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+block_size, j:j+block_size]

def minCutPath(errors):
    pq = [(error, [i]) for i, error in enumerate(errors[0])]
    heapq.heapify(pq)
    h, w = errors.shape
    seen = set()
    while pq:
        error, path = heapq.heappop(pq)
        d = len(path)
        idx = path[-1]
        if d == h:
            return path
        for delta in (-1, 0, 1):
            ni = idx + delta
            if 0 <= ni < w and (d, ni) not in seen:
                heapq.heappush(pq, (error + errors[d, ni], path + [ni]))
                seen.add((d, ni))

def minCutPatch(patch, block_size, overlap, res, y, x):
    patch = patch.copy()
    dy, dx, _ = patch.shape
    minCut = np.zeros_like(patch, dtype=bool)
    if x > 0:
        left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
        leftL2 = np.sum(left**2, axis=2)
        for i, j in enumerate(minCutPath(leftL2)):
            minCut[i, :j] = True
    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
        upL2 = np.sum(up**2, axis=2)
        for j, i in enumerate(minCutPath(upL2.T)):
            minCut[:i, j] = True
    np.copyto(patch, res[y:y+dy, x:x+dx], where=minCut)
    return patch

def quilt(image_path, block_size, num_block):
    texture = Image.open(image_path)
    texture = util.img_as_float(texture)
    overlap = block_size // 6
    h = num_block[0]*block_size - (num_block[0]-1)*overlap
    w = num_block[1]*block_size - (num_block[1]-1)*overlap
    res = np.zeros((h, w, texture.shape[2]))
    for i in range(num_block[0]):
        for j in range(num_block[1]):
            y = i*(block_size-overlap)
            x = j*(block_size-overlap)
            patch = randomBestPatch(texture, block_size, overlap, res, y, x)
            patch = minCutPatch(patch, block_size, overlap, res, y, x)
            res[y:y+block_size, x:x+block_size] = patch
    return (res*255).astype(np.uint8)

# =========================================================
# MAIN API (SAFE & COMPLETE)
# =========================================================

def get_creased(
    input_file,
    output_directory,
    ifWrinkles=False,
    ifCreases=False,
    crease_angle=0,
    num_creases_vertically=3,
    num_creases_horizontally=2,
    bbox=False
):
    img = cv2.imread(input_file).astype("float32") / 255.0
    hh, ww = img.shape[:2]

    # ---------------- Wrinkles ----------------
    if ifWrinkles:
        wrinkle_file = random.choice(list(WRINKLES_DIR.iterdir()))
        wrinklesImg = quilt(str(wrinkle_file), 250, (1, 1))
        wrinklesImg = cv2.cvtColor(wrinklesImg, cv2.COLOR_BGR2GRAY)
        wrinklesImg = wrinklesImg.astype("float32") / 255.0
        wrinkles = cv2.resize(wrinklesImg, (ww, hh))
        shift = np.mean(wrinkles) - 0.4
        wrinkles = cv2.subtract(wrinkles, shift)

    # ---------------- Creases ----------------
    if ifCreases:
        creases = np.ones((hh, ww), dtype=np.float32)
        cv2.line(creases, (0, hh//2), (ww, hh//2), 1.25, 5)
        folds = cv2.GaussianBlur(creases, (3,3), 0)
        folds = cv2.cvtColor(folds, cv2.COLOR_GRAY2BGR)
        img *= folds

    # ---------------- Apply wrinkles ----------------
    if ifWrinkles:
        t = cv2.cvtColor(wrinkles, cv2.COLOR_GRAY2BGR)
        thresh = cv2.threshold(wrinkles, 0.6, 1, cv2.THRESH_BINARY)[1]
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        low = 2 * img * t
        high = 1 - 2*(1-img)*(1-t)
        img = low*(1-thresh) + high*thresh

    img = (255*img).clip(0,255).astype(np.uint8)
    cv2.imwrite(input_file, img)
    return input_file