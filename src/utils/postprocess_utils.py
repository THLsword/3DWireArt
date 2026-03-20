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
from scipy.spatial import ConvexHull
import networkx as nx
from .curve_utils import bezier_sample
from sklearn.decomposition import PCA
import alphashape

def curve_2_pcd_kchamfer(x1, x2, k):
    b = x1.shape[0]
    x1 = x1.view(b, -1, 3)
    x1_norm = x1.pow(2).sum(-1, keepdim=True) # [8, 4374, 1]
    x2_norm = x2.pow(2).sum(-1, keepdim=True) # [8, 4096, 1]

    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-10).sqrt_()

    x, x_idx = torch.topk(res, k, dim=2, largest=False)
    return x, x_idx

def get_unique_curve(curve_idx):
    sorted_curve_idx, _ = curve_idx.sort()
    sorted_curve_idx = np.array(sorted_curve_idx)
    unique_curve, unique_idx = np.unique(sorted_curve_idx, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    curve_idx = curve_idx[unique_idx]
    return curve_idx

def project_curve_to_pcd(curves, pcd_points, batch_size, sample_num, k):
    # curves     [curve_num, cp_num(4), 3]
    # pcd_points [b, 4096, 3]

    # find k points close to curves' sample points
    curve_num = curves.shape[0]
    curves = curves.repeat(batch_size, 1, 1, 1) # (b, curve_num, cp_num, 3)
    linspace = torch.linspace(0, 1, sample_num).to(curves).flatten()[..., None]
    curve_points = bezier_sample(linspace, curves)
    # _, pcd_idx, _, _ = curve_chamfer(curve_points, pcd_points)
    _, pdc_k_idx = curve_2_pcd_kchamfer(curve_points, pcd_points, k) # in losses.py
    pcd_idx = torch.unique(pdc_k_idx)
    sampled_pcd = pcd_points[0][pcd_idx] # (n, 3) only used to save as obj

    # each curves' points idx and cood
    review_idx = pdc_k_idx.view(curve_num, -1, k) # (1, 13440, k) -> (96, 140, k)
    curve_idx_list = []  # (96, n)
    curve_cood_list = [] # (96, n, 3)
    for i in review_idx:
        curve_idx_list.append(torch.unique(i))
        cood = pcd_points[0][torch.unique(i)]
        curve_cood_list.append(cood)

    ####
    # sampled_pcd: (n, 3) only used to save as obj
    # review_idx : (96, 140, k) index of pcd 
    # curve_idx_list: (96, n) each curve's unique idx of pcd
    # curve_cood_list: (96, n, 3)
    #### 

    return sampled_pcd, review_idx, curve_idx_list, curve_cood_list

def curve_topology(vertex_idx, curve_idx, curves, curves_mask):
    # curve [n, n]: idx of connected curves
    # (96, 4) -> (96, 2) head and tail
    curve_ht_idx = curve_idx[:,[0,-1]] # (96,2)
    topology_mask = []
    for i in curve_ht_idx:
        mask1 = (curve_ht_idx == i[0]).sum(1) & curves_mask
        mask2 = (curve_ht_idx == i[1]).sum(1) & curves_mask
        different_value_mask = ~(mask1 == mask2)
        match1 = (mask1 & different_value_mask).sum() 
        match2 = (mask2 & different_value_mask).sum() 
        if match1 == 0 or match2 == 0:
            topology_mask.append(False)
        else:
            topology_mask.append(True)
    topology_mask = torch.tensor(topology_mask)

    return topology_mask

def delete_single_curve(G):
    while True:
        single_list = []
        for edge in list(G.edges):
            l = edge[0]
            r = edge[1]
            if G.degree[l]==1 or G.degree[r]==1:
                single_list.append(edge)
        for i in single_list:
            G.remove_edge(i[0],i[1])
            # print("remove edge", i[0],i[1])
        if len(single_list)==0:
            break
    return G

def PCA_of_curve(curve):
    pca = PCA(n_components=3)
    pca.fit(curve)
    transformed_data = pca.transform(curve)
    # 获取主成分方向
    components = pca.components_
    unit_vector = components[0]
    start_point = curve.mean(axis=0)
    end_point = start_point + unit_vector
    return start_point, end_point

def find_deletable_edges(G):
    graph_list = []
    removed_edges = []
    for edge in list(G.edges):
        bool_append = False
        G_copy = G.copy()
        l = edge[0]
        r = edge[1]
        if G_copy.degree[l]==3 or G_copy.degree[r]==3:
            G_copy.remove_edge(l, r)
            bool_append = True
        G_copy = delete_single_curve(G_copy)
        # detect if the curve has been deleted once
        for i in removed_edges:
            if not G_copy.has_edge(i[0],i[1]):
                bool_append = False
        if bool_append:
            removed_edges.append(edge)
            graph_list.append(G_copy)

    return graph_list

def create_curve_graph(curve_idx, curve_thresh_mask):
    curve_idx = np.array(curve_idx)
    G = nx.Graph()
    # add edges into G
    for i, curve in enumerate(curve_idx):
        if curve_thresh_mask[i]:
            G.add_edge(curve[0],curve[-1])
            G.edges[curve[0],curve[-1]]['idx']=i

    # delete single curves(curves with 1 degree end points)
    G = delete_single_curve(G)

    graph_list = find_deletable_edges(G)

    return G, graph_list

def get_rotate_matrix(start_point, end_point):
    # rotate to xy plane
    theta1 = np.arctan2(start_point[2], start_point[0])
    R_1 = np.array([
        [np.cos(theta1), 0, np.sin(theta1)],
        [0, 1, 0],
        [-np.sin(theta1), 0, np.cos(theta1)]
    ])
    rotated_start_point = np.dot(R_1, start_point)
    end_point = np.dot(R_1, end_point)

    # rotate to zy plane
    theta2 = np.arctan2(rotated_start_point[0], rotated_start_point[1])
    R_2 = np.array([
        [np.cos(theta2), -np.sin(theta2), 0],
        [np.sin(theta2), np.cos(theta2), 0],
        [0, 0, 1]
    ])
    rotated_start_point = np.dot(R_2, rotated_start_point)
    end_point = np.dot(R_2, end_point)

    # end_point rotate to xy plane
    theta3 = np.arctan2(end_point[2], end_point[0])
    R_3 = np.array([
        [np.cos(theta3), 0, np.sin(theta3)],
        [0, 1, 0],
        [-np.sin(theta3), 0, np.cos(theta3)]
    ])

    return np.dot(R_3, np.dot(R_2, R_1))

def render(pcd, rotate_matrix, image_size):
    counter = 0
    areas = []
    for i in rotate_matrix:
        rotated_pcd = np.dot(i, pcd.T).T
        projection = rotated_pcd[:, 1:3]
        img = np.zeros((image_size, image_size))
        img_points = []
        for point in projection:
            x_idx = int(point[0] * (image_size - 1)/2)+int(image_size/2)
            y_idx = int(point[1] * (image_size - 1)/2)+int(image_size/2)
            img[x_idx, y_idx] = 1
            img_points.append([x_idx, y_idx])
            # if y_idx+1 < image_size and x_idx+1 < image_size:
            #     img[y_idx+1, x_idx+1] = 1 
        
        # # test & save image
        # img = img*255
        # img = (img).astype(np.uint8)
        # image = Image.fromarray(img)
        # image.save(f"test{counter}.png")
        # counter = counter+1

        # compute area
        y, x = np.where(img > 0)
        points_2d = list(zip(x, y))

        alpha_shape_pcd = alphashape.alphashape(points_2d, 0.2)
        area = alpha_shape_pcd.area

        # img_points = np.array(img_points)
        # hull = ConvexHull(img_points)
        # area = hull.volume

        areas.append(area)
    
    return np.array(areas)

def compute_IOU(rotate_matrix, G, graph_list, bsplines):
    # bsplines [curve_num, sample_num, 3] -> [48, 28, 3]
    image_size = 128
    G_edge_idx = []
    for i in list(G.edges):
        G_edge_idx.append(G.edges[i[0], i[1]]['idx'])
    # G_edge_idx = set(G_edge_idx)
    G_curves = bsplines[G_edge_idx].reshape((-1,3))
    G_area = render(G_curves, rotate_matrix, image_size)

    max_IOUs = []
    for i in tqdm.tqdm(graph_list):
        i_edge_idx = []
        for j in list(i.edges):
            i_edge_idx.append(i.edges[j[0], j[1]]['idx'])
        i_curves = bsplines[i_edge_idx].reshape((-1,3))
        i_area = render(i_curves, rotate_matrix, image_size)
        max_IOU = np.max(G_area - i_area)
        max_IOUs.append(max_IOU)
    max_IOUs = np.array(max_IOUs)
    min_idx = np.argmin(max_IOUs)
    min_IOU = np.min(max_IOUs)
    print("max_IOUs: ", max_IOUs)
    
    newgraph_list = find_deletable_edges(graph_list[min_idx])
    return graph_list[min_idx], newgraph_list, min_IOU