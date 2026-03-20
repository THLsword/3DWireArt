import os
import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
from einops import rearrange, repeat
from PIL import Image
from scipy.interpolate import BSpline
import networkx as nx
from sklearn.decomposition import PCA

from dataset.load_pcd import load_npz, load_obj
from dataset.load_template import load_template
from utils.patch_utils import *
from utils.losses import *
from utils.curve_utils import * 
from utils.mview_utils import multiview_sample, curve_probability
from utils.postprocess_utils import get_unique_curve, project_curve_to_pcd, delete_single_curve, create_curve_graph, find_deletable_edges, compute_IOU
from utils.save_data import save_img, save_pcd_obj, save_curves

def create_bspline(mean_curve_points):
    print("mean_curve_points:", mean_curve_points.shape)
    k = 3
    sample_num = 50
    curves = np.zeros([mean_curve_points.shape[0], sample_num, 3])
    for i, curve in enumerate(mean_curve_points):
        # curve (n, 3) 
        # rangeeduce number of sampling points
        head = curve[0, :]
        tail = curve[-1, :]
        middle = curve[1:-1:2, :]
        curve = torch.vstack((head, middle, tail))
        # bspline
        curve = curve.numpy()
        _, idxs = np.unique(curve, axis=0, return_index=True)
        unique_curve = curve[np.sort(idxs)]
        cp_num = unique_curve.shape[0]
        k = int(cp_num/3)
        m = k + cp_num + 1
        t = np.linspace(0, 1, m-2*k)
        t = np.concatenate(([t[0]]*k, t, [t[-1]]*k))
        spl = BSpline(t, unique_curve, k)
        tnew = np.linspace(t[0], t[-1], sample_num)
        new_curve = spl(tnew) # (sample_num, 3)
        curves[i] = new_curve
    return curves

def post_processing(**kwargs):
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")

    # init param
    batch_size = kwargs["batch_size"]

    # load .npz point cloud
    model_path = kwargs['model_path']
    pcd_points, pcd_normals, pcd_area = load_npz(model_path)
    # pcd_points, pcd_normals, pcd_area = pcd_points.to(device), pcd_normals.to(device), pcd_area.to(device)
    pcd_points = pcd_points.repeat(batch_size,1,1)

    # load template
    template_path = kwargs['template_path']
    _, vertex_idx, face_idx, symmetriy_idx, curve_idx = load_template(template_path)
    vertex_idx, face_idx, curve_idx = vertex_idx.to(device), face_idx.to(device), curve_idx.to(device)
    sample_num = int(np.ceil(np.sqrt(4096/face_idx.shape[0])))*2

    # load multi view points
    mv_path = kwargs['mv_path']
    mv_points = load_obj(f"{mv_path}/multi_view.obj").to(device)

    # unique curves: curves were calculated twice
    curve_idx = get_unique_curve(curve_idx)

    # load control points -> curves 
    output_path = kwargs['output_path']
    control_points = load_obj(f"{output_path}/control_points.obj").to(device)
    curves = control_points[curve_idx] # (curve_num, cp_num, 3)
    # delete useless curve or not
    if os.path.exists(f'{output_path}/curves_mask.pt') and kwargs['d_curve']:
        curves_mask = torch.load(f'{output_path}/curves_mask.pt')
        curves = curves[curves_mask]

    # project_curve_to_pcd
    ''' 
    sampled_pcd:     (n, 3) only used to save as obj
    review_idx :     (96, 140, k) index of pcd 
    curve_idx_list : (96, n) each curve's unique idx of pcd
    curve_cood_list: (96, n, 3)
    '''
    sampled_pcd, review_idx, curve_idx_list, curve_cood_list = project_curve_to_pcd(curves, pcd_points, batch_size, sample_num, kwargs['k'])
    print("project to point clod")

    # new matching method. 
    # [96, 140, k] <-match-> all mv points. 
    # [96, 140, k] -> [96, 140T/F]
    curves_points_idx = review_idx # (96, 140, k)
    new_rate = []
    for c_idx in curves_points_idx: # (140, k)
        review_c_idx = c_idx.view(-1) # (140*k)
        coods = pcd_points[0][review_c_idx] # (140*k, 3)
        # match 
        matches = coods.unsqueeze(1) == mv_points.unsqueeze(0)
        matches = matches.all(dim=2)
        matched_indices_tensor1 = matches.any(dim=1) # [curve_on_pcd num]
        matched_indices_tensor2 = matches.any(dim=0) # [mv num]
        matched_indices_tensor1 = matched_indices_tensor1.view(-1, kwargs['k']).sum(1)
        mask = matched_indices_tensor1 > 0
        rate = mask.sum() / c_idx.shape[0]
        new_rate.append(rate)
    new_rate = torch.tensor(new_rate)
    curve_thresh_mask = new_rate > kwargs['match_rate']
    print("curve-multiview matching")

    # curve topology
    G, graph_list = create_curve_graph(curve_idx, curve_thresh_mask)
    topology_mask = torch.zeros(curve_idx.shape[0])
    for i in list(G.edges):
        topology_mask[G.edges[i[0],i[1]]['idx']]=1
    curve_thresh_mask_d = topology_mask.bool() # [96] or [48]

    # get pcd's PCA, Bsplines, and determine different views
    pcd_np = np.array(pcd_points[0])
    pca = PCA(n_components=3)
    pca.fit(pcd_np)
    all_bspline = create_bspline(pcd_points[0][review_idx].mean(dim=2)) # (48, 400, 3)
    transformed_data = pca.transform(pcd_np)
    transformed_bspline = pca.transform(np.array(all_bspline).reshape((-1,3)))
    transformed_bspline = transformed_bspline.reshape((all_bspline.shape))
    pca_x, pca_y, pca_z = np.max(transformed_data, axis=0) + np.abs(np.min(transformed_data, axis=0))
        # rotate matrix (will project to yz plane)
    rotate_y_angels = [0, np.arctan2(pca_z*1.5, pca_x), np.pi/2, np.pi-np.arctan2(pca_z*1.5, pca_x)]
    rotate_matrix = []
    for i in rotate_y_angels:
        matrix = np.array([
            [np.cos(i), 0, np.sin(i)],
            [0, 1, 0],
            [-np.sin(i), 0, np.cos(i)]
        ])
        rotate_matrix.append(matrix)
    rotate_matrix.append([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    rotate_matrix = np.stack(rotate_matrix)
        # compute IOU
    bool_delete = True
    while bool_delete:
        G , graph_list, min_IOU = compute_IOU(rotate_matrix, G, graph_list, transformed_bspline)
        if min_IOU > 1 - kwargs['IOU_thres']:
            bool_delete = False
        if len(graph_list) == 0:
            bool_delete = False
        # bool_delete = False

    curve_thresh_mask2 = torch.zeros_like(curve_thresh_mask)
    for i in list(G.edges):
        curve_thresh_mask2[G.edges[i[0], i[1]]['idx']] = True

    # ------------------------------------------------------------------------------------
    ## create/output new curves
    ## curves_points_idx = (96, 140, k), pcd_points = (n, 3)
    print("save objs")
    output_curves = pcd_points[0][curves_points_idx][curve_thresh_mask] # (96, 140, k, 3)
    mean_curve_points = output_curves.mean(dim=2) # (n_matched, 140, 3)
    bspline = create_bspline(mean_curve_points)
    save_pcd_obj(f"{output_path}/output_bspline0.obj", bspline.reshape(-1,3))
    
    
    output_curves = pcd_points[0][curves_points_idx][curve_thresh_mask_d] # (96, 140, k, 3)
    mean_curve_points = output_curves.mean(dim=2) # (96, 140, k, 3) -> (n_matched, 140, 3)
    bspline = create_bspline(mean_curve_points)
    save_pcd_obj(f"{output_path}/output_bspline1.obj", bspline.reshape(-1,3))

    # output_curves = pcd_points[0][curves_points_idx][curve_thresh_mask2] # (96, 140, k, 3)
    # mean_curve_points = output_curves.mean(dim=2) # (n_matched, 140, 3)
    # bspline = create_bspline(mean_curve_points)
    # save_obj(f"{output_path}/output_bspline2.obj", bspline.reshape(-1,3))

    # save_curves(f"{output_path}/output_curves.obj", mean_curve_points)
    save_pcd_obj(f"{output_path}/sampled_pcd.obj", sampled_pcd)

if __name__ == '__main__':
    # ` python src/train.py `
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="data/models/cat")
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")
    parser.add_argument('--output_path', type=str, default="output_cat_test")
    parser.add_argument('--mv_path', type=str, default="render_utils/train_outputs")

    parser.add_argument('--epoch', type=int, default="1")
    parser.add_argument('--batch_size', type=int, default="1") # 不要改，就是1
    parser.add_argument('--learning_rate', type=float, default="0.01")

    parser.add_argument('--d_curve', type=bool, default=False) # 是否删掉不需要的curve
    parser.add_argument('--k', type=int, default=5) # 裡curve採樣點最近的k個點
    parser.add_argument('--match_rate', type=float, default=0.4) 

    parser.add_argument('--IOU_thres', type=float, default=0.8) 


    args = parser.parse_args()
    post_processing(**vars(args)) # find curves in pcd