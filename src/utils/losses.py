import torch
from pytorch3d.ops.knn import knn_gather, knn_points


def area_weighted_chamfer_loss(
    mtds,    # [b, patch, sample_num]
    points,  # [b, patch, sample_num, 3]
    normals, # [b, patch, sample_num, 3]
    pcd_points, # [b, 4096, 3]
    target_normals,
    chamfer_weight_rate,
    multi_view_weights,
    compute_normals=True
):
    """ Compute area-weighted Chamfer loss. """

    b = points.shape[0]
    points = points.view(b, -1, 3)

    # find the distance of points to center point
    center_point = pcd_points.mean(dim=1) # [b, 3]
    distances, dis_to_center, gt_to_center = batched_cdist_l2(points, pcd_points, center_point) # distances [b, n_sample_points, n_mesh_points]

    # chamfer weights, The further away, the greater the weight
    mean_dis = dis_to_center.mean(1) # [b] # dis_to_center [b, n_sample_points]
    mean_gt_dis = gt_to_center.mean(1)     # gt_to_center  [b, n_mesh_points]
    distance_weight    = (torch.max(torch.tensor(0.).to(mtds),(dis_to_center-mean_dis).unsqueeze(1)) * 0.2 + torch.ones_like(dis_to_center).to(dis_to_center)).detach()
    distance_weight_gt = (torch.max(torch.tensor(0.).to(mtds),(gt_to_center-mean_gt_dis).unsqueeze(1)) * 0.2 + torch.ones_like(gt_to_center).to(gt_to_center)).detach()

    chamferloss_a, idx_a = distances.min(2)  # [b, n_sample_points]
    chamferloss_b, idx_b = distances.min(1)  # [b, n_mesh_points] 4096
    if multi_view_weights != None:
        chamferloss_b = chamferloss_b * multi_view_weights

    if compute_normals:
        normals = normals.view(b, -1, 3)

        # [b, n_sample_points, 1, 3]
        idx_a = idx_a[..., None, None].expand(-1, -1, -1, 3)
        nearest_target_normals = \
            target_normals[:, None].expand(list(distances.shape) + [3]) \
            .gather(index=idx_a, dim=2).squeeze(2)  # [b, n_sample_points, 3]

        # [b, 1, n_mesh_points, 3]
        idx_b = idx_b[..., None, :, None].expand(-1, -1, -1, 3)
        nearest_normals = \
            normals[:, :, None].expand(list(distances.shape) + [3]) \
            .gather(index=idx_b, dim=1).squeeze(1)  # [b, n_mesh_points, 3]

        normalsloss_a = torch.sum((nearest_target_normals - normals)**2, dim=-1)
        normalsloss_b = torch.sum((nearest_normals - target_normals)**2, dim=-1)
    else:
        normalsloss_a = torch.zeros_like(chamferloss_a)
        normalsloss_b = torch.zeros_like(chamferloss_b)

    mtds = mtds.view(b, -1)
    chamferloss_a = torch.sum(mtds*chamferloss_a*distance_weight, dim=-1) / mtds.sum(-1) # [b]
    chamferloss_b = (chamferloss_b*distance_weight_gt).mean(1) # [b]
    # chamferloss_a = torch.sum(mtds*chamferloss_a, dim=-1) / mtds.sum(-1) # [b]
    # chamferloss_b = (chamferloss_b).mean(1) # [b]
    chamfer_loss = ((chamferloss_a+chamferloss_b).mean() / 2).view(1)
    
    normalsloss_a = torch.sum(mtds*normalsloss_a, dim=-1) / mtds.sum(-1)
    normalsloss_b = normalsloss_b.mean(-1)
    normal_loss = ((normalsloss_a+normalsloss_b).mean() / 2).view(1)

    return chamfer_loss, normal_loss

def warm_up_chamfer_loss(
    mtds, # [b, patch, cp]
    points, # [b, patch, cp, 3]
    pcd_points, # [b, 4096, 3]
    ):

    b = points.shape[0]
    points = points.view(b, -1, 3)

    # find the distance of points to center point
    center_point = pcd_points.mean(dim=1) # [b, 3]

    distances, dis_to_center, gt_to_center = batched_cdist_l2(points, pcd_points, center_point) # distances [b, n_sample_points, n_mesh_points]

    chamferloss_a, idx_a = distances.min(2)  # [b, n_sample_points]
    chamferloss_b, idx_b = distances.min(1)  # [b, n_mesh_points] 4096

    mtds = mtds.view(b, -1)
    chamferloss_a = torch.sum(mtds*chamferloss_a, dim=-1) / mtds.sum(-1) # [b]
    chamferloss_b = (chamferloss_b).mean(1) # [b]
    chamfer_loss = ((chamferloss_a+chamferloss_b).mean() / 2).view(1)

    return chamfer_loss


def planar_patch_loss(st, points, mtds):
    """Compute planar patch loss from control points, samples, and Jacobians.

    st -- [..., 2] // (n, u, v)
    points -- [..., 3]
    """

    X = torch.cat([st.new_ones(list(st.shape[:-1]) + [1]), st], dim=-1)
    b = torch.inverse(X.transpose(-1, -2) @ X) @ X.transpose(-1, -2) @ points
    distances = (X @ b - points).pow(2).sum(-1)

    # planar_loss = torch.sum(distances, dim=-1)
    planar_loss = torch.sum(distances*mtds, dim=-1) / mtds.sum(-1)

    return planar_loss

def patch_symmetry_loss(xs, ys, vertices):
    """ Compute the symmetry loss to make the pairs of the control points symmetric to the reflection plane.
    Issue: see Issues 2 in README.md.
    """

    xs_ = vertices[:, xs]
    ys_ = vertices[:, ys] * vertices.new_tensor([[[-1, 1, 1]]])
    symmetry_loss = torch.sum((xs_ - ys_)**2, dim=-1).mean()
    return symmetry_loss

def kl_divergence_loss(mean, variance):
    """ Compute the KL-divergence loss for VAE. """

    kld_loss = torch.mean(-0.5 * torch.sum(1 + variance - mean ** 2 - variance.exp(), dim = 1), dim = 0)
    
    return kld_loss

def patch_overlap_loss(mtds, gt_area):
    """ Compute overlap loss to ensure the curve conciseness. """

    total_area = mtds.mean(dim=2).sum(dim=1)
    loss = torch.max(torch.tensor(0.).to(mtds), total_area - gt_area).pow(2).mean()
    # print(torch.max(torch.tensor(0.).to(mtds), total_area - gt_area).shape)
    return loss

def curve_perpendicular_loss(patches):
    """ Compute the tangent vector of the curves to ensure the orthogonality. """

    t = torch.empty(
        patches.shape[0], 
        patches.shape[1],
        2
    ).to(patches)
    t[..., 0] = 0.
    t[..., 1] = 1.

    t = t[..., None]

    sides = [patches[..., :4, :], patches[..., 3:7, :],
             patches[..., 6:10, :], patches[..., [9, 10, 11, 0], :]]

    tangent_v = [
        cal_tangent_vector(t, sides[0]), # (b, patch#, 3)
        cal_tangent_vector(t, sides[1]), 
        cal_tangent_vector(t, sides[2]), 
        cal_tangent_vector(t, sides[3]), 
    ]

    cos = torch.nn.CosineSimilarity(dim=-1)
    loss = torch.abs(cos(tangent_v[0][..., 0, :], tangent_v[3][..., 1, :]))
    loss = loss + torch.abs(cos(tangent_v[0][..., 1, :], tangent_v[1][..., 0, :]))
    loss = loss + torch.abs(cos(tangent_v[1][..., 1, :], tangent_v[2][..., 0, :]))
    loss = loss + torch.abs(cos(tangent_v[2][..., 1, :], tangent_v[3][..., 0, :]))

    loss = (loss / 4).mean()
    return loss

def curve_perpendicular_loss_8(patches):
    """ Compute the tangent vector of the curves to ensure the orthogonality. """

    t = torch.empty(
        patches.shape[0], 
        patches.shape[1],
        2
    ).to(patches)
    t[..., 0] = 0.
    t[..., 1] = 1.

    t = t[..., None]

    sides = [patches[..., :8, :], patches[..., 7:15, :],
             patches[..., 14:22, :], patches[..., [21, 22, 23, 24, 25, 26, 27, 0], :]] 

    tangent_v = [
        cal_tangent_vector_8(t, sides[0]), # (b, patch#, 3)
        cal_tangent_vector_8(t, sides[1]), 
        cal_tangent_vector_8(t, sides[2]), 
        cal_tangent_vector_8(t, sides[3]), 
    ]

    cos = torch.nn.CosineSimilarity(dim=-1)
    loss = torch.abs(cos(tangent_v[0][..., 0, :], tangent_v[3][..., 1, :]))
    loss = loss + torch.abs(cos(tangent_v[0][..., 1, :], tangent_v[1][..., 0, :]))
    loss = loss + torch.abs(cos(tangent_v[1][..., 1, :], tangent_v[2][..., 0, :]))
    loss = loss + torch.abs(cos(tangent_v[2][..., 1, :], tangent_v[3][..., 0, :]))

    loss = (loss / 4).mean()
    return loss

def cal_tangent_vector_8(t, params):
    A = 3 * params.new_tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                    [-7, 7, 0, 0, 0, 0, 0, 0],
                    [21, -42, 21, 0, 0, 0, 0, 0],
                    [-35, 105, -105, 35, 0, 0, 0, 0],
                    [35, -140, 210, -140, 35, 0, 0, 0],
                    [-21, 105, -210, 210, -105, 21, 0, 0],
                    [7, -42, 105, -140, 105, -42, 7, 0],
                    [-1, 7, -21, 35, -35, 21, -7, 1]
                    ])
    dim = params.new_tensor([0, 1, 2, 3, 4, 5, 6, 7]) 
    dt = t.pow(dim) @ A @ params
    tangent_v = dt / torch.sqrt(torch.sum(dt**2, dim=-1, keepdims=True))
    
    return tangent_v

def cal_tangent_vector(t, params):
    A = 3 * params.new_tensor([[-1, 1, 0, 0],
                                [2, -4, 2, 0],
                                [-1, 3, -3, 1],
                                [0, 0, 0, 0]])
    dim = params.new_tensor([0, 1, 2, 3]) 
    dt = t.pow(dim) @ A @ params
    tangent_v = dt / torch.sqrt(torch.sum(dt**2, dim=-1, keepdims=True))
    
    return tangent_v

def collision_detection_loss(
    grid_points,
    sigma,
    grid_point_edges,
    triangulation,
    triangulation_edges,
    adjacent_pairs,
    non_adjacent_pairs,
    point_to_triangle_distance
):
    b, _, _, _ = grid_points.shape

    collision_loss = grid_points.new_zeros([b, 0])
    triangles = grid_points[:, :, triangulation]

    i1s, i2s, e1s = zip(*adjacent_pairs)
    points_i = grid_points[:, i1s]
    point_idxs = torch.tensor(
                    [grid_point_edges[e] for e in e1s]
                ).to(points_i.device)
    point_idxs = point_idxs[None, :, :, None].expand(b, -1, -1, 3)
    points_i = torch.gather(points_i, 2, point_idxs)
    points_j = grid_points[:, i2s]

    triangles_i = triangles[:, i1s]
    triangle_idxs = torch.tensor(
                        [triangulation_edges[e] for e in e1s]
                    ).to(triangles_i.device)
    triangle_idxs = triangle_idxs[None, :, :, None, None].expand(b, -1, -1, 3, 3)
    triangles_i = torch.gather(triangles_i, 2, triangle_idxs)
    triangles_j = triangles[:, i2s]

    idxs = bboxes_intersect(
        points_i, points_j, dim=2
    ).any(0).nonzero().squeeze(1)
    n_adjacent_intersections = idxs.shape[0]

    if n_adjacent_intersections > 0:
        points_i = points_i[:, idxs].view([-1] + list(points_i.shape[2:]))
        points_j = points_j[:, idxs].view([-1] + list(points_j.shape[2:]))
        triangles_i = triangles_i[:, idxs].view([-1] + list(triangles_i.shape[2:]))
        triangles_j = triangles_j[:, idxs].view([-1] + list(triangles_j.shape[2:]))
        d1 = point_to_triangle_distance(points_i, triangles_j)
        d2 = point_to_triangle_distance(points_j, triangles_i)

        d = torch.min(d1, d2).view(b, -1)
        collision_loss = torch.cat(
            [collision_loss, torch.exp(-(d/sigma)**2)],
            dim=1
        )

    i1s, i2s = zip(*non_adjacent_pairs)
    idxs = bboxes_intersect(
        grid_points[:, i1s], grid_points[:, i2s], dim=2
    ).any(0).nonzero().squeeze(1)
    n_nonadjacent_intersections = idxs.shape[0]
    i1s = torch.tensor(i1s).to(grid_points.device)[idxs]
    i2s = torch.tensor(i2s).to(grid_points.device)[idxs]

    if n_nonadjacent_intersections > 0:
        points_i = grid_points[:, i1s].view(
            [-1] + list(grid_points.shape[2:]))
        points_j = grid_points[:, i2s].view(
            [-1] + list(grid_points.shape[2:]))
        triangles_i = triangles[:, i1s].view(
            [-1] + list(triangles.shape[2:]))
        triangles_j = triangles[:, i2s].view(
            [-1] + list(triangles.shape[2:]))
        d1 = point_to_triangle_distance(points_i, triangles_j)
        d2 = point_to_triangle_distance(points_j, triangles_i)
        d = torch.min(d1, d2).view(b, -1)
        collision_loss = torch.cat(
            [collision_loss, torch.exp(-(d/sigma)**2)],
            dim=1
        )

    del triangles

    if n_adjacent_intersections + n_nonadjacent_intersections > 0:
        collision_loss = collision_loss.sum(-1).mean()
    else:
        collision_loss = torch.zeros_like(1)
    
    return collision_loss

def template_normal_loss(mtds, normals, template_normals):
    normal_loss = torch.sum((normals - template_normals)**2, dim=-1)
    normal_loss = torch.sum(mtds*normal_loss, dim=-1) / mtds.sum(-1)
    
    return normal_loss

def batched_cdist_l2(x1, x2, x3):
    """Compute batched l2 cdist."""
    # x3 is the center point of point cloud
    x1_norm = x1.pow(2).sum(-1, keepdim=True) # [8, 4374, 1]
    x2_norm = x2.pow(2).sum(-1, keepdim=True) # [8, 4096, 1]
    
    x3 = x3.unsqueeze(1) # [b, 1, 3]
    x3_norm = x3.pow(2).sum(-1, keepdim=True) # [8, 1]

    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-10).sqrt_()
    # ||(x1 - x2)|| = ((x1 ^ 2) - 2 * (x1 * x2) + (x2 ^ 2)) ^ (1/2)
    # [b, n_sample_points, n_mesh_points]

    # distance of sample points to center point 
    res2 = torch.baddbmm(
        x1_norm.transpose(-2, -1),
        x3,
        x1.transpose(-2, -1),
        alpha=-2
    ).add_(x3_norm).clamp_min_(1e-10).sqrt_()

    # distance of gt points to center point 
    res3 = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x3,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x3_norm).clamp_min_(1e-10).sqrt_()
    
    return res, res2.squeeze(1), res3.squeeze(1)

class PointToTriangleDistance(torch.autograd.Function):
    """Autograd function for computing smallest point to triangle distance."""

    @staticmethod
    def forward(ctx, points, triangles):
        """Compute smallest distance between each point and triangle batch.

        points -- [batch_size, n_points, 3]
        triangles -- [batch_size, n_triagles, 3, 3]
        """
        b = points.shape[0]

        v21 = triangles[:, None, :, 1]-triangles[:, None, :, 0]
        v32 = triangles[:, None, :, 2]-triangles[:, None, :, 1]
        v13 = triangles[:, None, :, 0]-triangles[:, None, :, 2]
        p1 = points[:, :, None] - triangles[:, None, :, 0]
        p2 = points[:, :, None] - triangles[:, None, :, 1]
        p3 = points[:, :, None] - triangles[:, None, :, 2]
        nor = torch.cross(v21, v13, dim=-1)

        cond = dot(torch.cross(v21, nor, dim=-1), p1).sign() \
            + dot(torch.cross(v32, nor, dim=-1), p2).sign() \
            + dot(torch.cross(v13, nor, dim=-1), p3).sign() < 2
        cond = cond.float()
        result = cond * torch.stack([
            dot2(v21 * torch.clamp(dot(v21, p1) / dot2(v21), 0, 1) - p1),
            dot2(v32 * torch.clamp(dot(v32, p2) / dot2(v32), 0, 1) - p2),
            dot2(v13 * torch.clamp(dot(v13, p3) / dot2(v13), 0, 1) - p3)
        ], dim=-1).min(-1)[0] + (1-cond) \
            * dot(nor, p1) * dot(nor, p1) / dot2(nor)
        result = result.squeeze(-1)

        _, nearest_tris_idxs = result.min(-1)  # [b, n_points]
        _, nearest_points_idxs = result.min(-2)  # [b, n_tris]
        ctx.save_for_backward(
            points, triangles, nearest_tris_idxs, nearest_points_idxs)

        return result.view(b, -1).min(-1)[0]

    @staticmethod
    def backward(ctx, grad_output):
        """Only consider the closest point-triangle pair for gradient."""
        points, triangles, nearest_tris_idxs, nearest_points_idxs = \
            ctx.saved_tensors
        grad_points = grad_tris = None

        if ctx.needs_input_grad[0]:
            idx = nearest_tris_idxs[..., None, None].expand(
                list(nearest_tris_idxs.shape) + [3, 3])
            nearest_tris = triangles.gather(index=idx, dim=1)
            with torch.enable_grad():
                distance = d_points_to_tris(points, nearest_tris)
                grad_points = torch.autograd.grad(outputs=distance, inputs=points,
                                               grad_outputs=grad_output,
                                               only_inputs=True)[0]
        if ctx.needs_input_grad[1]:
            idx = nearest_points_idxs[..., None].expand(
                list(nearest_points_idxs.shape) + [3])
            nearest_points = points.gather(index=idx, dim=1)
            with torch.enable_grad():
                distance = d_points_to_tris(nearest_points, triangles)
                grad_tris = torch.autograd.grad(outputs=distance,
                                             inputs=triangles,
                                             grad_outputs=grad_output,
                                             only_inputs=True)[0]

        return grad_points, grad_tris

def d_points_to_tris(points, triangles):
    """Compute distance frome each point to the corresponding triangle.

    points -- [b, n, 3]
    triangles -- [b, n, 3, 3]
    """
    v21 = triangles[:, :, 1]-triangles[:, :, 0]
    v32 = triangles[:, :, 2]-triangles[:, :, 1]
    v13 = triangles[:, :, 0]-triangles[:, :, 2]
    p1 = points - triangles[:, :, 0]
    p2 = points - triangles[:, :, 1]
    p3 = points - triangles[:, :, 2]
    nor = torch.cross(v21, v13, dim=-1)

    cond = dot(torch.cross(v21, nor, dim=-1), p1).sign() \
        + dot(torch.cross(v32, nor, dim=-1), p2).sign() \
        + dot(torch.cross(v13, nor, dim=-1), p3).sign() < 2
    cond = cond.float()
    result = cond * torch.stack([
        dot2(v21 * torch.clamp(dot(v21, p1) / dot2(v21), 0, 1) - p1),
        dot2(v32 * torch.clamp(dot(v32, p2) / dot2(v32), 0, 1) - p2),
        dot2(v13 * torch.clamp(dot(v13, p3) / dot2(v13), 0, 1) - p3)
    ], dim=-1).min(-1)[0] + (1-cond) * dot(nor, p1) * dot(nor, p1) / dot2(nor)
    return result.squeeze(-1).min(-1)[0]

def dot(a, b):
    """Dot product."""
    return torch.sum(a*b, dim=-1, keepdim=True)

def dot2(a):
    """Squared norm."""
    return dot(a, a)

def bboxes_intersect(points1, points2, dim=1):
    """Compute whether bounding boxes of two point clouds intersect."""
    
    min1 = points1.min(dim)[0]
    max1 = points1.max(dim)[0]
    min2 = points2.min(dim)[0]
    max2 = points2.max(dim)[0]
    center1 = (min1 + max1)/2
    center2 = (min2 + max2)/2
    size1 = max1 - min1
    size2 = max2 - min2
    
    return ((center1 - center2).abs() * 2 <= size1 + size2).all(-1)

def flatness_area_loss(st, points, mtds):
    # area
    area = mtds.mean(dim=2) # (b, p)
    area = torch.tensor(area.shape[1]).to(mtds) * area

    # flatness
    X = torch.cat([st.new_ones(list(st.shape[:-1]) + [1]), st], dim=-1)
    b = torch.inverse(X.transpose(-1, -2) @ X) @ X.transpose(-1, -2) @ points
    distances = (X @ b - points).pow(2).sum(-1) # (b, p, distances)
    # copy flatness without grad
    distances_copy = distances.detach()
    mtds_copy = mtds.detach()
    distances_copy = torch.sum(distances_copy*mtds_copy, dim=-1) / mtds_copy.sum(-1) # (b, p)

    distance_mean = distances_copy.mean(dim=1).unsqueeze(1).expand_as(distances_copy)
    weight1 = torch.max(torch.tensor(0.).to(mtds),(distances_copy - distance_mean)) # (b, p)
    weight2 = torch.max(torch.tensor(0.).to(mtds),(distance_mean - distances_copy))
    # print(weight1.shape)

    loss = weight2 / area
    # loss = torch.tensor(1.).to(mtds)/torch.sum(weight2*area, dim = -1).mean()
    
    return loss

def multiview_curve_chamfer_loss(x1, x2):
    # x1 is curve sample points
    # x2 is mesh sample points
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
    # ||(x1 - x2)|| = ((x1 ^ 2) - 2 * (x1 * x2) + (x2 ^ 2)) ^ (1/2)
    # [b, n_sample_points, n_mesh_points]

    chamferloss_a, idx_a = res.min(2)  # [b, n_sample_points]
    chamferloss_b, idx_b = res.min(1)  # [b, n_mesh_points]

    # chamferloss = (chamferloss_a.mean(1) + chamferloss_b.mean(1))/2
    chamferloss = chamferloss_b.mean(1)
    return chamferloss

def curve_chamfer(x1, x2):
    # x1 is curve sample points
    # x2 is mesh sample points
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
    # ||(x1 - x2)|| = ((x1 ^ 2) - 2 * (x1 * x2) + (x2 ^ 2)) ^ (1/2)
    # [b, n_sample_points, n_mesh_points]

    chamferloss_a, idx_a = res.min(2)  # [b, x1]
    chamferloss_b, idx_b = res.min(1)  # [b, x2]

    return chamferloss_a, idx_a, chamferloss_b, idx_b

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

def curve_curvature_loss(curves, linspace):
    # Reduce curvature and make curves straighter
    t = linspace.repeat(1,3) # (16,3)

    p0 = curves[0][:,0:1, :]
    p1 = curves[0][:,1:2, :]
    p2 = curves[0][:,2:3, :]
    p3 = curves[0][:,3:4, :]
    
    B = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    # First derivative
    B_prime = 3 * (1 - t)**2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t**2 * (p3 - p2)
    # Second derivative
    B_double_prime = 6 * (1 - t) * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)
    # curvature
    cross_product = torch.cross(B_prime, B_double_prime)
    numerator = torch.norm(cross_product, dim=2)
    denominator =torch.norm(B_prime, dim=2)**3 + 0.00001
    curvature = numerator / denominator # (96,16)

    # weighted curvature
    weights = torch.tensor([((i-0.5)**8)*64 for i in linspace]).to(curves)
    curvature = (curvature * weights).mean()

    return curvature


def compute_beam_gap_loss(points, normals, pcd_points, thres):
    # view [b, n, 3]
    batch_size = points.shape[0]
    points = points.view(batch_size, -1, 3)
    normals = normals.view(batch_size, -1, 3)
    # KNN
    pk12 = knn_points(points, pcd_points, K=3).idx[0]
    pk21 = knn_points(pcd_points, points, K=3).idx[0]
    loop = pk21[pk12].view(pk12.shape[0], -1)
    knn_mask = (loop == torch.arange(0, pk12.shape[0], device=points.device)[:, None]).sum(dim=1) > 0

    points = points[0]
    pcd_points = pcd_points[0]
    normals = normals[0]
    normals = normals[~ knn_mask, :]
    masked_points = points[~ knn_mask, :]
    displacement = masked_points[:, None, :] - pcd_points[:, :3]
    distance = displacement.norm(dim=-1)

    # filter vectors with angles less than a threshold
    mask = (torch.abs(torch.sum((displacement / distance[:, :, None]) * normals[:, None, :], dim=-1)) > thres)
    dmin, argmin = distance.min(dim=-1)
    distance_no_inf = distance.clone()
    distance_no_inf[~mask] = float('inf')
    dmin, argmin = distance_no_inf.min(dim=-1)

    non_inf_mask = ~torch.isinf(dmin)
    loss = dmin[non_inf_mask].mean()

    return loss