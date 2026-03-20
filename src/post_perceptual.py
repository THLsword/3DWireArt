import os
import torch
import numpy as np
import argparse
from PIL import Image, ImageDraw
import networkx as nx
from scipy.interpolate import BSpline
import alphashape
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time

from dataset.load_pcd import load_npz, load_obj
from dataset.load_template import load_template
from utils.patch_utils import *
from utils.losses import *
from utils.curve_utils import *
from utils.postprocess_utils import get_unique_curve, project_curve_to_pcd
from utils.create_mesh import create_mesh
from utils.save_data import save_pcd_obj

def create_bspline(mean_curve_points):
    print("mean_curve_points:", mean_curve_points.shape)
    k = 3
    sample_num = 50
    curves = np.zeros([mean_curve_points.shape[0], sample_num, 3])
    for i, curve in enumerate(mean_curve_points):
        # curve: (n, 3)
        # Reduce the number of sampling points before fitting B-spline.
        head = curve[0, :]
        tail = curve[-1, :]
        middle = curve[1:-1:2, :]
        curve = torch.vstack((head, middle, tail))
        # B-spline fitting.
        curve = curve.numpy()
        _, idxs = np.unique(curve, axis=0, return_index=True)
        unique_curve = curve[np.sort(idxs)]
        cp_num = unique_curve.shape[0]
        k = min(3, cp_num - 1)
        m = k + cp_num + 1
        t = np.linspace(0, 1, m - 2 * k)
        t = np.concatenate(([t[0]] * k, t, [t[-1]] * k))
        spl = BSpline(t, unique_curve, k)
        tnew = np.linspace(t[0], t[-1], sample_num)
        new_curve = spl(tnew)  # (sample_num, 3)
        curves[i] = new_curve
    return curves

def render(data):
    """
    Render curves and compute alpha shape.

    inputs: dict ['bspline_remian', 'image_size', 'i: int', 'alpha_value', 'save_img: bool']
    outputs: alpha-shape area and perimeter length
    alpha_value 越小，轮廓越大，细节越少
    """
    rotated_pcd = data['bspline_remian']
    image_size = data['image_size']
    num = data['i']  # for saving PNG
    alpha_value = data['alpha_value']
    save_img = data['save_img']

    # Projection and rasterization.
    projection = rotated_pcd[:, 1:3]

    img = np.zeros((image_size, image_size))
    for point in projection:
        x_idx = int(point[0] * (image_size - 1) / 2) + int(image_size / 2)
        y_idx = int(point[1] * (image_size - 1) / 2) + int(image_size / 2)
        img[x_idx, y_idx] = 1
        if y_idx + 1 < image_size and x_idx + 1 < image_size:
            img[x_idx + 1, y_idx + 1] = 1
    del projection

    # Compute area and perimeter.
    y, x = np.where(img > 0)
    points_2d = list(zip(x, y))
    alpha_shape_pcd = alphashape.alphashape(points_2d, alpha_value)
    area = alpha_shape_pcd.area
    length = alpha_shape_pcd.length

    if save_img:
        output_path = "render_results"
        os.makedirs(output_path, exist_ok=True)
        # Save rendered image.
        image = Image.fromarray((img * 255).astype(np.uint8))
        image.save(f"{output_path}/render{num}.png")

        # Save polygon image.
        image = Image.new("1", (image_size, image_size), 0)  # "1" 表示二值化模式，0 表示黑色
        draw = ImageDraw.Draw(image)
        draw.polygon(list(alpha_shape_pcd.exterior.coords), outline=1, fill=1)
        image.save(f"{output_path}/polygon{num}.png")

    return area, length

def create_graph(curve_idx):
    curve_idx = np.array(curve_idx)
    G = nx.Graph()
    # Add edges into G.
    for i, curve in enumerate(curve_idx):
        G.add_edge(curve[0], curve[-1])
        G.edges[curve[0], curve[-1]]['idx'] = [i]
    return G

def graph_delete_curve(G, idx):
    G_ = G.copy()
    # idx is the curve index list stored on the edge.
    for edge in list(G_.edges):
        if G_.edges[edge[0], edge[1]]['idx'] == idx:
            G_.remove_edge(edge[0], edge[1])
            break
    # # if node's degree=2, merge 2 curves
    # for node in list(G_.nodes):
    #     if G_.degree(node) == 2:
    #         for i,value in G_.adjacency():
    #             if i==node:
    #                 adj = []    # len=2
    #                 adj_idx = []# len=2
    #                 for j in value:
    #                     adj.append(j)
    #                     for k in value[j]['idx']:
    #                         adj_idx.append(k)
    #                 for j in adj:
    #                     G_.remove_edge(j,i)
    #                 G_.add_edge(adj[0],adj[1])
    #                 G_.edges[adj[0],adj[1]]['idx']=adj_idx
    for node in list(G_.nodes):
        if G_.degree[node] == 0:
            G_.remove_node(node)
    return G_

def graph_curve_removable(G, idx):
    G_ = G.copy()
    # # detect if one degree is 3
    # for edge in list(G_.edges):
    #     if G_.edges[edge[0],edge[1]]['idx'] == idx:
    #         if G_.degree[edge[0]]==3 or G_.degree[edge[1]]==3:
    #             break
    #         else:
    #             return False
    # Detect whether deleting this curve disconnects the graph.
    for edge in list(G_.edges):
        if G_.edges[edge[0], edge[1]]['idx'] == idx:
            G_.remove_edge(edge[0], edge[1])
            break
    # for edge in list(G_.edges):
    #     if G_.degree(edge[0]) == 1 or G_.degree(edge[1]) == 1:
    #         return False
    for node in list(G_.nodes):
        if G_.degree[node] == 0:
            G_.remove_node(node)
    g_num = len(list(nx.connected_components(G_)))
    if g_num > 1:
        return False
    else:
        return True

def build_rotation_matrices():
    """Create the fixed view set used for multi-view silhouette scoring."""
    rotate_y_angles = [-np.pi * 0.33, 0.0, np.pi * 0.33, np.pi / 2, np.pi - np.pi * 0.33]
    rotate_matrices = [
        np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        for angle in rotate_y_angles
    ]
    rotate_matrices.append(np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ]))
    return np.stack(rotate_matrices, axis=0)

def collect_curve_indices(G, excluded_idx=None):
    """Flatten the curve ids stored on graph edges, optionally skipping one edge."""
    curve_indices = []
    for start_node, end_node in G.edges:
        edge_curve_indices = G.edges[start_node, end_node]['idx']
        if edge_curve_indices == excluded_idx:
            continue
        curve_indices.extend(edge_curve_indices)
    return curve_indices

def compute_multiview_areas(bspline_points, rotate_matrices, image_size, alpha_value, executor):
    """Render all predefined views and return the alpha-shape area for each view."""
    rotated_bsplines = np.stack([bspline_points @ matrix.T for matrix in rotate_matrices], axis=0)
    render_inputs = [
        {
            'bspline_remian': points,
            'image_size': image_size,
            'i': view_idx,
            'alpha_value': alpha_value,
            'save_img': False,
        }
        for view_idx, points in enumerate(rotated_bsplines)
    ]
    return np.array([area for area, _ in executor.map(render, render_inputs)])

def export_curve_mesh(curves, curve_indices, output_path, object_curve_num, radius):
    """Sample the remaining Bezier curves and export them as a mesh."""
    bezier_t = torch.linspace(0, 1, 16).to(curves).flatten()[..., None]
    remaining_curve_cps = torch.stack([curves[idx] for idx in curve_indices])
    remaining_curve_points = bezier_sample(bezier_t, remaining_curve_cps)
    create_mesh(remaining_curve_points, radius, output_path, object_curve_num)

def training(**kwargs):
    """Prune redundant curves by multi-view alpha-shape loss while preserving connectivity."""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device("cuda:0"))

    # Load the target point cloud and the template topology used to index curves.
    model_path = kwargs['model_path']
    output_path = kwargs['output_path']
    post_output_path = os.path.join(output_path, 'post_outputs')
    os.makedirs(post_output_path, exist_ok=True)
    pcd_points, _ = load_npz(model_path)
    batch_size = kwargs['batch_size']
    _, _, face_idx, _, curve_idx = load_template(kwargs['template_path'])
    sample_num = int(np.ceil(np.sqrt(4096 / face_idx.shape[0])))

    # Remove duplicate template curves and keep graph indices aligned with the optional mask.
    curve_idx = get_unique_curve(curve_idx)
    control_points = load_obj(f"{output_path}/control_points.obj")
    curves = control_points[curve_idx]  # (curve_num, cp_num, 3)
    curves_mask_path = f'{output_path}/curves_mask.pt'
    if os.path.exists(curves_mask_path) and kwargs['d_curve']:
        curves_mask = torch.load(curves_mask_path)
        curve_idx = curve_idx[curves_mask]
        curves = curves[curves_mask]
    G = create_graph(curve_idx)

    # Associate each fitted curve with nearby point-cloud support before perceptual pruning.
    sampled_pcd, review_idx, _, _ = project_curve_to_pcd(
        curves,
        pcd_points.repeat(batch_size, 1, 1),
        batch_size,
        sample_num,
        kwargs['k'],
    )
    print("project to point cloud")
    save_pcd_obj(f"{post_output_path}/sampled_pcd.obj", sampled_pcd)

    # Smooth the projected support points and evaluate them from a fixed set of views.
    all_bspline = create_bspline(pcd_points[review_idx].mean(dim=2))
    rotate_matrices = build_rotation_matrices()
    image_size = 128
    alpha_value = kwargs['alpha_value']
    object_curve_num = kwargs['object_curve_num']
    mv_thresh = kwargs['mv_thresh']

    with ProcessPoolExecutor(max_workers=len(rotate_matrices)) as executor:
        area_global = compute_multiview_areas(
            all_bspline.reshape((-1, 3)),
            rotate_matrices,
            image_size,
            alpha_value,
            executor,
        )

        # Greedily delete the curve that least changes the global silhouettes.
        while object_curve_num < len(collect_curve_indices(G)):
            delete_idx = None
            min_iou_loss = float('inf')
            area_denom = np.maximum(area_global, 1e-8)

            for edge in tqdm(list(G.edges)):
                edge_curve_indices = G.edges[edge[0], edge[1]]['idx']
                if not graph_curve_removable(G, edge_curve_indices):
                    continue

                remaining_curve_indices = collect_curve_indices(G, excluded_idx=edge_curve_indices)
                candidate_areas = compute_multiview_areas(
                    all_bspline[remaining_curve_indices].reshape((-1, 3)),
                    rotate_matrices,
                    image_size,
                    alpha_value,
                    executor,
                )
                iou_loss_global = np.max((area_global - candidate_areas) / area_denom)
                if iou_loss_global < min_iou_loss:
                    min_iou_loss = iou_loss_global
                    delete_idx = edge_curve_indices

            if delete_idx is None:
                print("No removable curve remains; stop pruning.")
                break
            if min_iou_loss > mv_thresh:
                print("min_IOU_loss:", min_iou_loss, " > mv_thresh")
                break

            G = graph_delete_curve(G, delete_idx)
            remaining_curve_indices = collect_curve_indices(G)
            print("min IOU loss: ", min_iou_loss)
            print("curves remain : ", len(remaining_curve_indices))

            if object_curve_num < 35 and len(remaining_curve_indices) == 35:
                export_curve_mesh(curves, remaining_curve_indices, post_output_path, 35, 0.003)

    remaining_curve_indices = collect_curve_indices(G)
    print('object_curve_num: ', object_curve_num)

    # Export the final remaining curves for downstream inspection and fabrication.
    export_curve_mesh(curves, remaining_curve_indices, post_output_path, object_curve_num, 0.002)
        
if __name__ == '__main__':
    start_time = time.time()
    model = 'tower'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=f"data/models/{model}")
    parser.add_argument('--template_path', type=str, default="data/templates/cube24")
    parser.add_argument('--output_path', type=str, default=f"outputs/{model}")
    parser.add_argument('--batch_size', type=int, default=1)  # Keep this as 1.
    parser.add_argument('--d_curve', type=bool, default=False)  # Drop masked curves if needed.
    parser.add_argument('--k', type=int, default=10)  # Number of nearest points per curve sample.
    parser.add_argument('--alpha_value', type=float, default=0.25)
    parser.add_argument('--object_curve_num', type=int, default=48)
    parser.add_argument('--mv_thresh', type=float, default=0.10)

    args = parser.parse_args()
    training(**vars(args))
    end_time = time.time()
    print(f"time: {(end_time - start_time):.4f}")
