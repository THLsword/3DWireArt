import argparse
import os
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.save_data import save_img, save_pcd_obj
from visual_prior_utils.img_alphashape import multi_process_image
from visual_prior_utils.pcd_renderer import pcd_renderer
from visual_prior_utils.visual_training import visual_training

MAX_ALPHA_WORKERS = 4


def main(args):
    # Prefer GPU when available because both rendering and optimization are tensor-heavy.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    data_dir = args.DATA_DIR
    filename = args.FILENAME
    view_angles = args.VIEW_ANGELS
    pcd_path = os.path.join(data_dir, filename)

    for output_dir in (args.RENDER_SAVE_DIR, args.ALPHA_SAVE_DIR, args.TRAIN_SAVE_DIR):
        os.makedirs(output_dir, exist_ok=True)

    with np.load(pcd_path) as npz_file:
        pointcloud = npz_file["points"]

    print("Start rendering pointcloud")
    rendered_images = pcd_renderer(pointcloud, view_angles, device)
    for index, rendered_image in enumerate(rendered_images):
        image = Image.fromarray(rendered_image)
        save_filename = f"{os.path.splitext(filename)[0]}_{index}.png"
        image.save(os.path.join(args.RENDER_SAVE_DIR, save_filename))

    print("Start computing alpha shape")
    alpha_size = args.ALPHA_SIZE
    expand_size = args.EXPAND_SIZE
    process_num = min(len(rendered_images), MAX_ALPHA_WORKERS)
    params_list = [(image, alpha_size, expand_size) for image in rendered_images]
    with Pool(processes=process_num) as pool:
        contour_imgs = list(
            tqdm(
                pool.imap(multi_process_image, params_list),
                total=len(params_list),
                desc="Processing",
            )
        )

    for index, contour_img in enumerate(contour_imgs):
        image = Image.fromarray(contour_img)
        save_filename = f"{os.path.splitext(filename)[0]}_{index}.png"
        image.save(os.path.join(args.ALPHA_SAVE_DIR, save_filename))

    contour_imgs = np.stack(contour_imgs)

    print("Start visual training")
    images, colors = visual_training(
        pointcloud,
        contour_imgs,
        args.EPOCH,
        view_angles,
        device,
        point_num=pointcloud.shape[0],
    )

    images = images.detach().cpu()
    colors = colors.detach().cpu()
    for index, image in enumerate(images):
        save_img(image.numpy(), f"{args.TRAIN_SAVE_DIR}/output_{index}.png")

    torch.save(colors, f"{args.TRAIN_SAVE_DIR}/weights.pt")
    # Keep only points whose learned visibility weight passes the binary threshold.
    mask = colors > 0.5
    masked_pcd = pointcloud[mask]
    save_pcd_obj(f"{args.TRAIN_SAVE_DIR}/multi_view.obj", masked_pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default="./data/models/cat")
    parser.add_argument('--RENDER_SAVE_DIR', type=str, default="./prep_outputs/render_outputs")
    parser.add_argument('--ALPHA_SAVE_DIR', type=str, default="./prep_outputs/alpha_outputs")
    parser.add_argument('--TRAIN_SAVE_DIR', type=str, default="./prep_outputs/train_outputs")
    parser.add_argument('--FILENAME', type=str, default="model_normalized_4096.npz")

    parser.add_argument('--EPOCH', type=int, default=50)
    parser.add_argument('--VIEW_ANGELS', type=float, nargs='+', default=[45, 90, 135, 225, 270, 315])
    parser.add_argument('--ALPHA_SIZE', type=float, default=50.0)
    parser.add_argument('--EXPAND_SIZE', type=int, default=1)

    args = parser.parse_args()

    main(args)
