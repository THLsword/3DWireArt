import os
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import numpy as np
import argparse
from einops import rearrange, repeat
from PIL import Image, ImageDraw
import networkx as nx
from scipy.interpolate import BSpline
import alphashape
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import psutil
import gc
from tqdm import tqdm
import time

from dataset.load_pcd import load_npz, load_obj
from dataset.load_template import load_template
from utils.patch_utils import *
from utils.losses import *
from utils.curve_utils import *
from utils.postprocess_utils import get_unique_curve, project_curve_to_pcd, delete_single_curve, create_curve_graph, find_deletable_edges, compute_IOU
from utils.create_mesh import create_mesh
from utils.graph_utils import minimum_path_coverage
from utils.save_data import save_img, save_pcd_obj, save_curves

from model.model_interface import Model

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

def training(**kwargs):
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        
    # Load .npz point cloud.
    model_path = kwargs['model_path']
    output_path = kwargs['output_path']
    pcd_points, pcd_normals = load_npz(model_path)
    # pcd_points, pcd_normals, pcd_area = pcd_points.to(device), pcd_normals.to(device), pcd_area.to(device)
    pcd_maen = abs(pcd_points).mean(0)  # [3]

    # Load template.
    batch_size = kwargs['batch_size']
    template_path = kwargs['template_path']
    template_params, vertex_idx, face_idx, symmetriy_idx, curve_idx = load_template(template_path)
    # template_params, vertex_idx, face_idx, symmetriy_idx, curve_idx =\
    # template_params.to(device), vertex_idx.to(device), face_idx.to(device), symmetriy_idx.to(device), curve_idx.to(device)
    
    sample_num = int(np.ceil(np.sqrt(4096 / face_idx.shape[0])))
    # Resize template to be similar in size.
    template_mean = abs(template_params.view(-1, 3)).mean(0)  # [3]
    template_params = (template_params.view(-1, 3) / template_mean * pcd_maen)
    template_params = template_params.repeat(batch_size, 1, 1)

    # Remove duplicated curves: curves were calculated twice.
    curve_idx = get_unique_curve(curve_idx)
    G = create_graph(curve_idx)

    # Load control points -> curves.
    output_path = kwargs['output_path']
    control_points = load_obj(f"{output_path}/control_points.obj")
    curves = control_points[curve_idx]  # (curve_num, cp_num, 3)
    # Optionally delete unused curves.
    if os.path.exists(f'{output_path}/curves_mask.pt') and kwargs['d_curve']:
        curves_mask = torch.load(f'{output_path}/curves_mask.pt')
        curves = curves[curves_mask]

    ################### post-processing #################
    ################### post-processing #################
    # project_curve_to_pcd
    '''  (n, 3) only used to save as obj
    review_idx :     (curve_num, 140, k) index of pcd
    curve_cood_list: (curve_num, n, 3)
    '''
    sampled_pcd, review_idx, _, curve_cood_list = project_curve_to_pcd(curves, pcd_points.repeat(batch_size, 1, 1), batch_size, sample_num, kwargs['k'])
    print("project to point cloud")
    save_pcd_obj(f"{output_path}/sampled_pcd.obj", sampled_pcd)

    # Get B-splines and determine different views.
    all_bspline = create_bspline(pcd_points[review_idx].mean(dim=2))  # (48, 400, 3)

    # Rotation matrix (will project to yz plane).
    rotate_y_angels = [-np.pi * 0.33, 0.0, np.pi * 0.33, np.pi / 2, np.pi - np.pi * 0.33]
    rotate_matrix = []
    for i in rotate_y_angels:
        matrix = np.array([
            [np.cos(i), 0, np.sin(i)],
            [0, 1, 0],
            [-np.sin(i), 0, np.cos(i)]
        ])
        rotate_matrix.append(matrix)
    # 俯视视角
    rotate_matrix.append([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    rotate_matrix = np.stack(rotate_matrix)

    object_curve_num = kwargs['object_curve_num']

    # Render alpha shape before removing curves.
    image_size = 128
    alpha_value = kwargs['alpha_value']
    bspline_remian = all_bspline.reshape((-1, 3))
    # Rotate bspline.
    rotated_bsplines = []
    for mat in rotate_matrix:
        transformed = bspline_remian @ mat.T
        rotated_bsplines.append(transformed)
    rotated_bsplines = np.stack(rotated_bsplines, axis=0)

    data_list = []
    for i, value in enumerate(rotated_bsplines):
        data_list.append({'bspline_remian': value, 'image_size': image_size, 'i': i, 'alpha_value': alpha_value, 'save_img': False})
    with ProcessPoolExecutor(max_workers=len(rotate_matrix)) as executor:
        as_results = list(executor.map(render, data_list))  # list[(area, length), ...]
    area_before_delet = []  # Area before deleting.
    length_bd = []  # Length before deleting.
    for as_result in as_results:
        area_before_delet.append(as_result[0])
        length_bd.append(as_result[1])

    ## Start removing curves.
    mv_thresh = kwargs['mv_thresh']
    curves_ramian = curve_cood_list.copy()
    area_global = np.array(area_before_delet)  # Global baseline before removing.
    while True:
        if object_curve_num == 48:
            break
        delete_idx = []
        min_IOU_loss = 1
        for edge in tqdm(list(G.edges)):
            j = G.edges[edge[0],edge[1]]['idx'] # list
            # detect
            continue_bool = False
            if not graph_curve_removable(G, j):
                continue_bool = True
            if continue_bool:
                continue
            
            ## IOU area/length after deletion
            remain_idx_list = []  # idx list except j
            for k in list(G.edges):
                idx_list = G.edges[k[0],k[1]]['idx']
                if idx_list != j:
                    for l in idx_list:
                        remain_idx_list.append(l)
            bspline_remian = all_bspline[remain_idx_list].reshape((-1, 3))
            data_list = []
            area_ad = []
            length_ad = []
            # Rotate bspline.
            rotated_bsplines = []
            for mat in rotate_matrix:
                transformed = bspline_remian @ mat.T
                rotated_bsplines.append(transformed)
            rotated_bsplines = np.stack(rotated_bsplines, axis=0)
            for i, value in enumerate(rotated_bsplines):
                data_list.append({'bspline_remian': value, 'image_size': image_size, 'i': i, 'alpha_value': alpha_value, 'save_img': False})
            with ProcessPoolExecutor(max_workers=len(rotate_matrix)) as executor:
                as_results = list(executor.map(render, data_list))
            for as_result in as_results:
                area_ad.append(as_result[0])
                length_ad.append(as_result[1])

            IOU_loss_global = max((area_global - np.array(area_ad)) / area_global)
            if IOU_loss_global < min_IOU_loss:
                min_IOU_loss = IOU_loss_global
                delete_idx = j

        if min_IOU_loss > mv_thresh and min_IOU_loss != 1:
            print("min_IOU_loss:", min_IOU_loss," > mv_thresh")
            break
        # Delete the selected curve from the graph.
        G = graph_delete_curve(G, delete_idx)
        print("min IOU loss: ", min_IOU_loss)
        # Break condition.
        remain_idx_list = []
        for k in list(G.edges):
            idx_list = G.edges[k[0],k[1]]['idx']
            for l in idx_list:
                remain_idx_list.append(l)
        print("curves remain : ", len(remain_idx_list))

        if object_curve_num < 35 and len(remain_idx_list) == 35:
            linspace = torch.linspace(0, 1, 16).to(curves).flatten()[..., None]
            curves_ramian_cps = torch.stack([curves[k] for k in remain_idx_list]) # [curve_num, cp_num, 3]
            curves_ramian_points = bezier_sample(linspace, curves_ramian_cps) # [curve_num, sample_num, 3]
            create_mesh(curves_ramian_points, 0.003, output_path, 35)
        
        if len(remain_idx_list) <= object_curve_num:
            print("len(remain_idx_list) <= object_curve_num")
            break

    remain_idx_list = []
    for k in list(G.edges):
        idx_list = G.edges[k[0],k[1]]['idx']
        for l in idx_list:
            remain_idx_list.append(l)
    curves_ramian_tensor = torch.cat([curves_ramian[k] for k in remain_idx_list])
    bspline_remian = all_bspline[remain_idx_list].reshape((-1, 3))
    print('object_curve_num: ', object_curve_num)
    # create_mesh(all_bspline[remain_idx_list], 0.003, output_path)

    # Create mesh: template Bezier curves.
    linspace = torch.linspace(0, 1, 16).to(curves).flatten()[..., None]
    curves_ramian_cps = torch.stack([curves[k] for k in remain_idx_list]) # [curve_num, cp_num, 3]
    curves_ramian_points = bezier_sample(linspace, curves_ramian_cps) # [curve_num, sample_num, 3]
    create_mesh(curves_ramian_points, 0.002, output_path, object_curve_num)
    
    # save_pcd_obj(f"{output_path}/sampled_pcd_perceptual.obj", curves_ramian_tensor)
    # save_pcd_obj(f"{output_path}/spline_perceptual.obj", bspline_remian)
        
if __name__ == '__main__':
    start_time = time.time()
    model = 'tower'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=f"data/models/{model}")
    # parser.add_argument('--template_path', type=str, default="data/templates/sphere24")
    parser.add_argument('--template_path', type=str, default="data/templates/cube24")
    parser.add_argument('--output_path', type=str, default=f"outputs/{model}")
    parser.add_argument('--prep_output_path', type=str, default=f"outputs/{model}/prep_outputs/train_outputs")

    parser.add_argument('--epoch', type=int, default="201")
    parser.add_argument('--batch_size', type=int, default="1") # 不要改，就是1
    parser.add_argument('--learning_rate', type=float, default="0.0005")

    parser.add_argument('--d_curve', type=bool, default=False) # 是否删掉不需要的curve
    parser.add_argument('--k', type=int, default=10) # 裡curve採樣點最近的k個點
    parser.add_argument('--match_rate', type=float, default=0.2)
    parser.add_argument('--alpha_value', type=float, default=0.25)
    parser.add_argument('--object_curve_num', type=int, default=48)
    parser.add_argument('--mv_thresh', type=float, default=0.10)
    parser.add_argument('--crossattention', type=bool, default=True)

    args = parser.parse_args()
    training(**vars(args))
    end_time = time.time()
    print(f"time: {(end_time - start_time):.4f}")
