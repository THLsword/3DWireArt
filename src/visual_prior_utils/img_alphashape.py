import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image
from alpha_shapes import Alpha_Shaper
from alpha_shapes.boundary import get_boundaries
from matplotlib.path import Path
from tqdm import tqdm

MAX_PROCESSES = 4


def extract_points(image):
    """Extract foreground pixel coordinates from a rendered RGB image."""
    gray = np.mean(image, axis=-1)
    y_coords, x_coords = np.where(gray > 0)
    y_coords = image.shape[0] - 1 - y_coords
    return list(zip(x_coords, y_coords))


def _build_alpha_vertices(points_2d, alpha_size):
    """Build alpha-shape boundary vertices with a conservative fallback alpha."""
    shaper = Alpha_Shaper(points_2d)
    try:
        alpha_shape = shaper.get_shape(alpha=alpha_size)
    except TypeError:
        alpha_shape = shaper.get_shape(alpha=3.0)

    vertices = []
    for boundary in get_boundaries(alpha_shape):
        exterior = Path(boundary.exterior)
        holes = [Path(hole) for hole in boundary.holes]
        path = Path.make_compound_path(exterior, *holes)
        vertices.append(path.vertices)

    return np.concatenate(vertices)


def _load_images(data_dir):
    """Load input images once so the benchmark paths operate on the same data."""
    loaded_images = []
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith('.png'):
            file_path = os.path.join(data_dir, filename)
            with Image.open(file_path) as image:
                loaded_images.append(image.copy())
    return loaded_images


def img_alphashape(input_img, alpha_size, expand_size: int):
    assert expand_size >= 1, f"expand_size must be >= 1, but got {expand_size}"

    data = np.asarray(input_img)
    points_2d = extract_points(data)
    npvertices = _build_alpha_vertices(points_2d, alpha_size)

    # Draw the extracted contour on a black background.
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    for x, y in npvertices:
        neighbors = [
            (x + dx, y + dy)
            for dy in range(-expand_size, expand_size + 1)
            for dx in range(-expand_size, expand_size + 1)
        ]
        for nx, ny in neighbors:
            nx = int(nx)
            ny = int(ny)
            if 0 <= nx < data.shape[1] and 0 <= ny < data.shape[0]:
                if np.any(data[data.shape[0] - 1 - ny, nx] > 0):
                    img[data.shape[0] - 1 - ny, nx] = [255, 255, 255]

    return img


def multi_process_image(params):
    """Wrapper used by multiprocessing.Pool."""
    image, alpha_size, expand_size = params
    return img_alphashape(image, alpha_size, expand_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default="./render_utils/render_outputs")
    parser.add_argument('--SAVE_DIR', type=str, default="./render_utils/alpha_outputs")
    parser.add_argument('--ALPHA_SIZE', type=float, default=50.0)
    parser.add_argument('--EXPAND_SIZE', type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.SAVE_DIR, exist_ok=True)

    images = []
    start_time = time.time()
    loaded_images = _load_images(args.DATA_DIR)

    process_num = min(len(loaded_images), MAX_PROCESSES)
    with Pool(processes=process_num) as pool:
        params_list = [(img, args.ALPHA_SIZE, args.EXPAND_SIZE) for img in loaded_images]
        images = list(
            tqdm(
                pool.imap(multi_process_image, params_list),
                total=len(params_list),
                desc="Processing",
            )
        )
    end_time = time.time()
    print('time: ', end_time - start_time)

    start_time = time.time()
    for filename in tqdm(os.listdir(args.DATA_DIR)):
        if filename.endswith('.png'):
            file_path = os.path.join(args.DATA_DIR, filename)
            with Image.open(file_path) as image:
                contour_img = img_alphashape(image, args.ALPHA_SIZE, args.EXPAND_SIZE)
            images.append(contour_img)
            contour_img = Image.fromarray(contour_img)
            save_filename = f'{os.path.splitext(filename)[0]}.png'
            contour_img.save(os.path.join(args.SAVE_DIR, save_filename))
    end_time = time.time()
    print('time: ', end_time - start_time)
