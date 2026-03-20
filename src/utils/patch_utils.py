from xml.dom import INDEX_SIZE_ERR
import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange

def coons_points(s, t, patches) -> torch.Tensor:
    """Sample points from Coons patch defined by params at s, t values.

    params -- [..., 12, 3]
    """
    sides = [patches[..., :4, :], patches[..., 3:7, :],
             patches[..., 6:10, :], patches[..., [9, 10, 11, 0], :]]
    corners = [patches[..., [0], :], patches[..., [3], :],
               patches[..., [9], :], patches[..., [6], :]]

    s = s[..., None]
    t = t[..., None]
    B = corners[0] * (1-s) * (1-t) + corners[1] * s * (1-t) + \
        corners[2] * (1-s) * t + corners[3] * s * t  # [..., n_samples, 3]

    Lc = bezier_sample(s, sides[0]) * (1-t) + bezier_sample(1-s, sides[2]) * t
    Ld = bezier_sample(t, sides[1]) * s + bezier_sample(1-t, sides[3]) * (1-s)
    
    return Lc + Ld - B

def coons_points_8(s, t, patches):
    """Sample points from Coons patch defined by params at s, t values.

    params -- [..., 12, 3]
    """
    sides = [patches[..., :8, :], patches[..., 7:15, :],
             patches[..., 14:22, :], patches[..., [21, 22, 23, 24, 25, 26, 27, 0], :]] 
    corners = [patches[..., [0], :], patches[..., [7], :],
               patches[..., [21], :], patches[..., [14], :]]

    s = s[..., None]
    t = t[..., None]
    B = corners[0] * (1-s) * (1-t) + corners[1] * s * (1-t) + \
        corners[2] * (1-s) * t + corners[3] * s * t  # [..., n_samples, 3]

    Lc = bezier_sample_8(s, sides[0]) * (1-t) + bezier_sample_8(1-s, sides[2]) * t
    Ld = bezier_sample_8(t, sides[1]) * s + bezier_sample_8(1-t, sides[3]) * (1-s)
    
    return Lc + Ld - B

def coons_normals(s, t, patches):
    params = patches[...,None]
    normals = torch.stack([(params[...,0,2,:]*(s - 1)**3 - params[...,9,2,:]*(s - 1)**3 - s*(3*params[...,3,2,:]*(t - 1)**2 - 3*params[...,4,2,:]*(t - 1)**2 + 3*params[...,5,2,:]*t**2 - 3*params[...,6,2,:]*t**2 + 6*params[...,5,2,:]*t*(t - 1) - 3*params[...,4,2,:]*t*(2*t - 2)) + params[...,3,2,:]*s - params[...,6,2,:]*s + (s - 1)*(3*params[...,0,2,:]*(t - 1)**2 - 3*params[...,11,2,:]*(t - 1)**2 - 3*params[...,9,2,:]*t**2 + 3*params[...,10,2,:]*t**2 + 6*params[...,10,2,:]*t*(t - 1) - 3*params[...,11,2,:]*t*(2*t - 2)) - params[...,0,2,:]*(s - 1) + params[...,9,2,:]*(s - 1) - params[...,3,2,:]*s**3 + params[...,6,2,:]*s**3 - 3*params[...,1,2,:]*s*(s - 1)**2 + 3*params[...,2,2,:]*s**2*(s - 1) - 3*params[...,7,2,:]*s**2*(s - 1) + 3*params[...,8,2,:]*s*(s - 1)**2)*(params[...,0,1,:]*(t - 1)**3 - params[...,3,1,:]*(t - 1)**3 + t*(3*params[...,8,1,:]*(s - 1)**2 - 3*params[...,9,1,:]*(s - 1)**2 + 3*params[...,6,1,:]*s**2 - 3*params[...,7,1,:]*s**2 - 6*params[...,7,1,:]*s*(s - 1) + 3*params[...,8,1,:]*s*(2*s - 2)) - params[...,6,1,:]*t + params[...,9,1,:]*t + (t - 1)*(3*params[...,0,1,:]*(s - 1)**2 - 3*params[...,1,1,:]*(s - 1)**2 + 3*params[...,2,1,:]*s**2 - 3*params[...,3,1,:]*s**2 + 6*params[...,2,1,:]*s*(s - 1) - 3*params[...,1,1,:]*s*(2*s - 2)) - params[...,0,1,:]*(t - 1) + params[...,3,1,:]*(t - 1) + params[...,6,1,:]*t**3 - params[...,9,1,:]*t**3 + 3*params[...,4,1,:]*t*(t - 1)**2 - 3*params[...,5,1,:]*t**2*(t - 1) + 3*params[...,10,1,:]*t**2*(t - 1) - 3*params[...,11,1,:]*t*(t - 1)**2) - (params[...,0,1,:]*(s - 1)**3 - params[...,9,1,:]*(s - 1)**3 - s*(3*params[...,3,1,:]*(t - 1)**2 - 3*params[...,4,1,:]*(t - 1)**2 + 3*params[...,5,1,:]*t**2 - 3*params[...,6,1,:]*t**2 + 6*params[...,5,1,:]*t*(t - 1) - 3*params[...,4,1,:]*t*(2*t - 2)) + params[...,3,1,:]*s - params[...,6,1,:]*s + (s - 1)*(3*params[...,0,1,:]*(t - 1)**2 - 3*params[...,11,1,:]*(t - 1)**2 - 3*params[...,9,1,:]*t**2 + 3*params[...,10,1,:]*t**2 + 6*params[...,10,1,:]*t*(t - 1) - 3*params[...,11,1,:]*t*(2*t - 2)) - params[...,0,1,:]*(s - 1) + params[...,9,1,:]*(s - 1) - params[...,3,1,:]*s**3 + params[...,6,1,:]*s**3 - 3*params[...,1,1,:]*s*(s - 1)**2 + 3*params[...,2,1,:]*s**2*(s - 1) - 3*params[...,7,1,:]*s**2*(s - 1) + 3*params[...,8,1,:]*s*(s - 1)**2)*(params[...,0,2,:]*(t - 1)**3 - params[...,3,2,:]*(t - 1)**3 + t*(3*params[...,8,2,:]*(s - 1)**2 - 3*params[...,9,2,:]*(s - 1)**2 + 3*params[...,6,2,:]*s**2 - 3*params[...,7,2,:]*s**2 - 6*params[...,7,2,:]*s*(s - 1) + 3*params[...,8,2,:]*s*(2*s - 2)) - params[...,6,2,:]*t + params[...,9,2,:]*t + (t - 1)*(3*params[...,0,2,:]*(s - 1)**2 - 3*params[...,1,2,:]*(s - 1)**2 + 3*params[...,2,2,:]*s**2 - 3*params[...,3,2,:]*s**2 + 6*params[...,2,2,:]*s*(s - 1) - 3*params[...,1,2,:]*s*(2*s - 2)) - params[...,0,2,:]*(t - 1) + params[...,3,2,:]*(t - 1) + params[...,6,2,:]*t**3 - params[...,9,2,:]*t**3 + 3*params[...,4,2,:]*t*(t - 1)**2 - 3*params[...,5,2,:]*t**2*(t - 1) + 3*params[...,10,2,:]*t**2*(t - 1) - 3*params[...,11,2,:]*t*(t - 1)**2), (params[...,0,0,:]*(s - 1)**3 - params[...,9,0,:]*(s - 1)**3 - s*(3*params[...,3,0,:]*(t - 1)**2 - 3*params[...,4,0,:]*(t - 1)**2 + 3*params[...,5,0,:]*t**2 - 3*params[...,6,0,:]*t**2 + 6*params[...,5,0,:]*t*(t - 1) - 3*params[...,4,0,:]*t*(2*t - 2)) + params[...,3,0,:]*s - params[...,6,0,:]*s + (s - 1)*(3*params[...,0,0,:]*(t - 1)**2 - 3*params[...,11,0,:]*(t - 1)**2 - 3*params[...,9,0,:]*t**2 + 3*params[...,10,0,:]*t**2 + 6*params[...,10,0,:]*t*(t - 1) - 3*params[...,11,0,:]*t*(2*t - 2)) - params[...,0,0,:]*(s - 1) + params[...,9,0,:]*(s - 1) - params[...,3,0,:]*s**3 + params[...,6,0,:]*s**3 - 3*params[...,1,0,:]*s*(s - 1)**2 + 3*params[...,2,0,:]*s**2*(s - 1) - 3*params[...,7,0,:]*s**2*(s - 1) + 3*params[...,8,0,:]*s*(s - 1)**2)*(params[...,0,2,:]*(t - 1)**3 - params[...,3,2,:]*(t - 1)**3 + t*(3*params[...,8,2,:]*(s - 1)**2 - 3*params[...,9,2,:]*(s - 1)**2 + 3*params[...,6,2,:]*s**2 - 3*params[...,7,2,:]*s**2 - 6*params[...,7,2,:]*s*(s - 1) + 3*params[...,8,2,:]*s*(2*s - 2)) - params[...,6,2,:]*t + params[...,9,2,:]*t + (t - 1)*(3*params[...,0,2,:]*(s - 1)**2 - 3*params[...,1,2,:]*(s - 1)**2 + 3*params[...,2,2,:]*s**2 - 3*params[...,3,2,:]*s**2 + 6*params[...,2,2,:]*s*(s - 1) - 3*params[...,1,2,:]*s*(2*s - 2)) - params[...,0,2,:]*(t - 1) + params[...,3,2,:]*(t - 1) + params[...,6,2,:]*t**3 - params[...,9,2,:]*t**3 + 3*params[...,4,2,:]*t*(t - 1)**2 - 3*params[...,5,2,:]*t**2*(t - 1) + 3*params[...,10,2,:]*t**2*(t - 1) - 3*params[...,11,2,:]*t*(t - 1)**2) - (params[...,0,2,:]*(s - 1)**3 - params[...,9,2,:]*(s - 1)**3 - s*(3*params[...,3,2,:]*(t - 1)**2 - 3*params[...,4,2,:]*(t - 1)**2 + 3*params[...,5,2,:]*t**2 - 3*params[...,6,2,:]*t**2 + 6*params[...,5,2,:]*t*(t - 1) - 3*params[...,4,2,:]*t*(2*t - 2)) + params[...,3,2,:]*s - params[...,6,2,:]*s + (s - 1)*(3*params[...,0,2,:]*(t - 1)**2 - 3*params[...,11,2,:]*(t - 1)**2 - 3*params[...,9,2,:]*t**2 + 3*params[...,10,2,:]*t**2 + 6*params[...,10,2,:]*t*(t - 1) - 3*params[...,11,2,:]*t*(2*t - 2)) - params[...,0,2,:]*(s - 1) + params[...,9,2,:]*(s - 1) - params[...,3,2,:]*s**3 + params[...,6,2,:]*s**3 - 3*params[...,1,2,:]*s*(s - 1)**2 + 3*params[...,2,2,:]*s**2*(s - 1) - 3*params[...,7,2,:]*s**2*(s - 1) + 3*params[...,8,2,:]*s*(s - 1)**2)*(params[...,0,0,:]*(t - 1)**3 - params[...,3,0,:]*(t - 1)**3 + t*(3*params[...,8,0,:]*(s - 1)**2 - 3*params[...,9,0,:]*(s - 1)**2 + 3*params[...,6,0,:]*s**2 - 3*params[...,7,0,:]*s**2 - 6*params[...,7,0,:]*s*(s - 1) + 3*params[...,8,0,:]*s*(2*s - 2)) - params[...,6,0,:]*t + params[...,9,0,:]*t + (t - 1)*(3*params[...,0,0,:]*(s - 1)**2 - 3*params[...,1,0,:]*(s - 1)**2 + 3*params[...,2,0,:]*s**2 - 3*params[...,3,0,:]*s**2 + 6*params[...,2,0,:]*s*(s - 1) - 3*params[...,1,0,:]*s*(2*s - 2)) - params[...,0,0,:]*(t - 1) + params[...,3,0,:]*(t - 1) + params[...,6,0,:]*t**3 - params[...,9,0,:]*t**3 + 3*params[...,4,0,:]*t*(t - 1)**2 - 3*params[...,5,0,:]*t**2*(t - 1) + 3*params[...,10,0,:]*t**2*(t - 1) - 3*params[...,11,0,:]*t*(t - 1)**2), (params[...,0,1,:]*(s - 1)**3 - params[...,9,1,:]*(s - 1)**3 - s*(3*params[...,3,1,:]*(t - 1)**2 - 3*params[...,4,1,:]*(t - 1)**2 + 3*params[...,5,1,:]*t**2 - 3*params[...,6,1,:]*t**2 + 6*params[...,5,1,:]*t*(t - 1) - 3*params[...,4,1,:]*t*(2*t - 2)) + params[...,3,1,:]*s - params[...,6,1,:]*s + (s - 1)*(3*params[...,0,1,:]*(t - 1)**2 - 3*params[...,11,1,:]*(t - 1)**2 - 3*params[...,9,1,:]*t**2 + 3*params[...,10,1,:]*t**2 + 6*params[...,10,1,:]*t*(t - 1) - 3*params[...,11,1,:]*t*(2*t - 2)) - params[...,0,1,:]*(s - 1) + params[...,9,1,:]*(s - 1) - params[...,3,1,:]*s**3 + params[...,6,1,:]*s**3 - 3*params[...,1,1,:]*s*(s - 1)**2 + 3*params[...,2,1,:]*s**2*(s - 1) - 3*params[...,7,1,:]*s**2*(s - 1) + 3*params[...,8,1,:]*s*(s - 1)**2)*(params[...,0,0,:]*(t - 1)**3 - params[...,3,0,:]*(t - 1)**3 + t*(3*params[...,8,0,:]*(s - 1)**2 - 3*params[...,9,0,:]*(s - 1)**2 + 3*params[...,6,0,:]*s**2 - 3*params[...,7,0,:]*s**2 - 6*params[...,7,0,:]*s*(s - 1) + 3*params[...,8,0,:]*s*(2*s - 2)) - params[...,6,0,:]*t + params[...,9,0,:]*t + (t - 1)*(3*params[...,0,0,:]*(s - 1)**2 - 3*params[...,1,0,:]*(s - 1)**2 + 3*params[...,2,0,:]*s**2 - 3*params[...,3,0,:]*s**2 + 6*params[...,2,0,:]*s*(s - 1) - 3*params[...,1,0,:]*s*(2*s - 2)) - params[...,0,0,:]*(t - 1) + params[...,3,0,:]*(t - 1) + params[...,6,0,:]*t**3 - params[...,9,0,:]*t**3 + 3*params[...,4,0,:]*t*(t - 1)**2 - 3*params[...,5,0,:]*t**2*(t - 1) + 3*params[...,10,0,:]*t**2*(t - 1) - 3*params[...,11,0,:]*t*(t - 1)**2) - (params[...,0,0,:]*(s - 1)**3 - params[...,9,0,:]*(s - 1)**3 - s*(3*params[...,3,0,:]*(t - 1)**2 - 3*params[...,4,0,:]*(t - 1)**2 + 3*params[...,5,0,:]*t**2 - 3*params[...,6,0,:]*t**2 + 6*params[...,5,0,:]*t*(t - 1) - 3*params[...,4,0,:]*t*(2*t - 2)) + params[...,3,0,:]*s - params[...,6,0,:]*s + (s - 1)*(3*params[...,0,0,:]*(t - 1)**2 - 3*params[...,11,0,:]*(t - 1)**2 - 3*params[...,9,0,:]*t**2 + 3*params[...,10,0,:]*t**2 + 6*params[...,10,0,:]*t*(t - 1) - 3*params[...,11,0,:]*t*(2*t - 2)) - params[...,0,0,:]*(s - 1) + params[...,9,0,:]*(s - 1) - params[...,3,0,:]*s**3 + params[...,6,0,:]*s**3 - 3*params[...,1,0,:]*s*(s - 1)**2 + 3*params[...,2,0,:]*s**2*(s - 1) - 3*params[...,7,0,:]*s**2*(s - 1) + 3*params[...,8,0,:]*s*(s - 1)**2)*(params[...,0,1,:]*(t - 1)**3 - params[...,3,1,:]*(t - 1)**3 + t*(3*params[...,8,1,:]*(s - 1)**2 - 3*params[...,9,1,:]*(s - 1)**2 + 3*params[...,6,1,:]*s**2 - 3*params[...,7,1,:]*s**2 - 6*params[...,7,1,:]*s*(s - 1) + 3*params[...,8,1,:]*s*(2*s - 2)) - params[...,6,1,:]*t + params[...,9,1,:]*t + (t - 1)*(3*params[...,0,1,:]*(s - 1)**2 - 3*params[...,1,1,:]*(s - 1)**2 + 3*params[...,2,1,:]*s**2 - 3*params[...,3,1,:]*s**2 + 6*params[...,2,1,:]*s*(s - 1) - 3*params[...,1,1,:]*s*(2*s - 2)) - params[...,0,1,:]*(t - 1) + params[...,3,1,:]*(t - 1) + params[...,6,1,:]*t**3 - params[...,9,1,:]*t**3 + 3*params[...,4,1,:]*t*(t - 1)**2 - 3*params[...,5,1,:]*t**2*(t - 1) + 3*params[...,10,1,:]*t**2*(t - 1) - 3*params[...,11,1,:]*t*(t - 1)**2)], dim=-1)
    
    return F.normalize(normals, p=2, dim=-1)

def coons_partial_derivative(s, t, patches):
    A = s.new_tensor([[1, 0, 0, 0],
                      [-3, 3, 0, 0],
                      [3, -6, 3, 0],
                      [-1, 3, -3, 1]])
    s = s[..., None]
    t = t[..., None]
    sides = [patches[..., :4, :], patches[..., 3:7, :],
             patches[..., 6:10, :], patches[..., [9, 10, 11, 0], :]]
    corners = [patches[..., [0], :], patches[..., [3], :],
               patches[..., [9], :], patches[..., [6], :]]

    Lc_ds = torch.cat((torch.zeros_like(s),torch.ones_like(s), 2*s, 3*s**2), dim=3) @ A @ sides[0] * (1-t) +\
            torch.cat((torch.zeros_like(s),-torch.ones_like(s), 2*s - 2, -3*((1-s)**2)), dim=3) @ A @ sides[2] * t
    Ld_ds = torch.cat((t**0,t**1,t**2,t**3), dim=3) @ A @ sides[1] +\
            torch.cat(((1-t)**0,(1-t)**1,(1-t)**2,(1-t)**3), dim=3) @ A @ sides[3] * (-1)
    B_ds = -corners[0] * (1-t) + corners[1] * (1-t) - \
            corners[2] * t + corners[3] * t   # [..., n_samples, 3]
    coons_point_ds = Lc_ds + Ld_ds - B_ds

    Lc_dt = torch.cat((s**0,s**1,s**2,s**3), dim=3) @ A @ sides[0] * (-1) +\
            torch.cat(((1-s)**0,(1-s)**1,(1-s)**2,(1-s)**3), dim=3) @ A @ sides[2]
    Ld_dt = torch.cat((torch.zeros_like(t),torch.ones_like(t), 2*t, 3*t**2), dim=3) @ A @ sides[1] * s +\
            torch.cat((torch.zeros_like(t),-torch.ones_like(t), 2*t - 2, -3*((1-t)**2)), dim=3) @ A @ sides[3] * (1-s)
    B_dt = -corners[0] * (1-s)  - corners[1] * s + \
            corners[2] * (1-s) + corners[3] * s   # [..., n_samples, 3]
    coons_point_dt = Lc_dt + Ld_dt - B_dt

    return coons_point_ds, coons_point_dt

def coons_partial_derivative_8(s, t, patches):
    A = s.new_tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                    [-7, 7, 0, 0, 0, 0, 0, 0],
                    [21, -42, 21, 0, 0, 0, 0, 0],
                    [-35, 105, -105, 35, 0, 0, 0, 0],
                    [35, -140, 210, -140, 35, 0, 0, 0],
                    [-21, 105, -210, 210, -105, 21, 0, 0],
                    [7, -42, 105, -140, 105, -42, 7, 0],
                    [-1, 7, -21, 35, -35, 21, -7, 1]
                    ])
    s = s[..., None]
    t = t[..., None]
    sides = [patches[..., :8, :], patches[..., 7:15, :],
             patches[..., 14:22, :], patches[..., [21, 22, 23, 24, 25, 26, 27, 0], :]] 
    corners = [patches[..., [0], :], patches[..., [7], :],
               patches[..., [21], :], patches[..., [14], :]]

    Lc_ds = torch.cat((torch.zeros_like(s),torch.ones_like(s), 2*s, 3*s**2, 4*s**3, 5*s**4, 6*s**5, 7*s**6), dim=3) @ A @ sides[0] * (1-t) +\
            torch.cat((torch.zeros_like(s),-torch.ones_like(s), 2*s - 2, -3*((1-s)**2), -4*(1-s)**3, -5*(1-s)**4, -6*(1-s)**5, -7*(1-s)**6), dim=3) @ A @ sides[2] * t
    Ld_ds = torch.cat((t**0,t**1,t**2,t**3,t**4,t**5,t**6,t**7), dim=3) @ A @ sides[1] +\
            torch.cat(((1-t)**0,(1-t)**1,(1-t)**2,(1-t)**3,(1-t)**4,(1-t)**5,(1-t)**6,(1-t)**7), dim=3) @ A @ sides[3] * (-1)
    B_ds = -corners[0] * (1-t) + corners[1] * (1-t) - \
            corners[2] * t + corners[3] * t   # [..., n_samples, 3]
    coons_point_ds = Lc_ds + Ld_ds - B_ds

    Lc_dt = torch.cat((s**0,s**1,s**2,s**3,s**4,s**5,s**6,s**7), dim=3) @ A @ sides[0] * (-1) +\
            torch.cat(((1-s)**0,(1-s)**1,(1-s)**2,(1-s)**3,(1-s)**4,(1-s)**5,(1-s)**6,(1-s)**7), dim=3) @ A @ sides[2]
    Ld_dt = torch.cat((torch.zeros_like(t),torch.ones_like(t), 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6), dim=3) @ A @ sides[1] * s +\
            torch.cat((torch.zeros_like(t),-torch.ones_like(t), 2*t - 2, -3*((1-t)**2), -4*(1-t)**3, -5*(1-t)**4, -6*(1-t)**5, -7*(1-t)**6), dim=3) @ A @ sides[3] * (1-s)
    B_dt = -corners[0] * (1-s)  - corners[1] * s + \
            corners[2] * (1-s) + corners[3] * s   # [..., n_samples, 3]
    coons_point_dt = Lc_dt + Ld_dt - B_dt


    return coons_point_ds, coons_point_dt

def coons_normals_(s, t, patches, control_point_num = 4):
    # s,t (batch, face, 144, 1)
    if control_point_num == 4:
        coons_point_ds, coons_point_dt = coons_partial_derivative(s, t, patches)
    elif control_point_num == 8:
        coons_point_ds, coons_point_dt = coons_partial_derivative_8(s, t, patches)
    normals = torch.cross(coons_point_ds, coons_point_dt)

    return F.normalize(normals, p=2, dim=-1)

def coons_mtds(s, t, patches):
    # params [..., 12, 3]
    params = patches[...,None]
    det = ((params[...,0,0,:]*(s - 1)**3 - params[...,9,0,:]*(s - 1)**3 - s*(3*params[...,3,0,:]*(t - 1)**2 - 3*params[...,4,0,:]*(t - 1)**2 + 3*params[...,5,0,:]*t**2 - 3*params[...,6,0,:]*t**2 + 6*params[...,5,0,:]*t*(t - 1) - 3*params[...,4,0,:]*t*(2*t - 2)) + params[...,3,0,:]*s - params[...,6,0,:]*s + (s - 1)*(3*params[...,0,0,:]*(t - 1)**2 - 3*params[...,11,0,:]*(t - 1)**2 - 3*params[...,9,0,:]*t**2 + 3*params[...,10,0,:]*t**2 + 6*params[...,10,0,:]*t*(t - 1) - 3*params[...,11,0,:]*t*(2*t - 2)) - params[...,0,0,:]*(s - 1) + params[...,9,0,:]*(s - 1) - params[...,3,0,:]*s**3 + params[...,6,0,:]*s**3 - 3*params[...,1,0,:]*s*(s - 1)**2 + 3*params[...,2,0,:]*s**2*(s - 1) - 3*params[...,7,0,:]*s**2*(s - 1) + 3*params[...,8,0,:]*s*(s - 1)**2)**2 + (params[...,0,1,:]*(s - 1)**3 - params[...,9,1,:]*(s - 1)**3 - s*(3*params[...,3,1,:]*(t - 1)**2 - 3*params[...,4,1,:]*(t - 1)**2 + 3*params[...,5,1,:]*t**2 - 3*params[...,6,1,:]*t**2 + 6*params[...,5,1,:]*t*(t - 1) - 3*params[...,4,1,:]*t*(2*t - 2)) + params[...,3,1,:]*s - params[...,6,1,:]*s + (s - 1)*(3*params[...,0,1,:]*(t - 1)**2 - 3*params[...,11,1,:]*(t - 1)**2 - 3*params[...,9,1,:]*t**2 + 3*params[...,10,1,:]*t**2 + 6*params[...,10,1,:]*t*(t - 1) - 3*params[...,11,1,:]*t*(2*t - 2)) - params[...,0,1,:]*(s - 1) + params[...,9,1,:]*(s - 1) - params[...,3,1,:]*s**3 + params[...,6,1,:]*s**3 - 3*params[...,1,1,:]*s*(s - 1)**2 + 3*params[...,2,1,:]*s**2*(s - 1) - 3*params[...,7,1,:]*s**2*(s - 1) + 3*params[...,8,1,:]*s*(s - 1)**2)**2 + (params[...,0,2,:]*(s - 1)**3 - params[...,9,2,:]*(s - 1)**3 - s*(3*params[...,3,2,:]*(t - 1)**2 - 3*params[...,4,2,:]*(t - 1)**2 + 3*params[...,5,2,:]*t**2 - 3*params[...,6,2,:]*t**2 + 6*params[...,5,2,:]*t*(t - 1) - 3*params[...,4,2,:]*t*(2*t - 2)) + params[...,3,2,:]*s - params[...,6,2,:]*s + (s - 1)*(3*params[...,0,2,:]*(t - 1)**2 - 3*params[...,11,2,:]*(t - 1)**2 - 3*params[...,9,2,:]*t**2 + 3*params[...,10,2,:]*t**2 + 6*params[...,10,2,:]*t*(t - 1) - 3*params[...,11,2,:]*t*(2*t - 2)) - params[...,0,2,:]*(s - 1) + params[...,9,2,:]*(s - 1) - params[...,3,2,:]*s**3 + params[...,6,2,:]*s**3 - 3*params[...,1,2,:]*s*(s - 1)**2 + 3*params[...,2,2,:]*s**2*(s - 1) - 3*params[...,7,2,:]*s**2*(s - 1) + 3*params[...,8,2,:]*s*(s - 1)**2)**2)*((params[...,0,0,:]*(t - 1)**3 - params[...,3,0,:]*(t - 1)**3 + t*(3*params[...,8,0,:]*(s - 1)**2 - 3*params[...,9,0,:]*(s - 1)**2 + 3*params[...,6,0,:]*s**2 - 3*params[...,7,0,:]*s**2 - 6*params[...,7,0,:]*s*(s - 1) + 3*params[...,8,0,:]*s*(2*s - 2)) - params[...,6,0,:]*t + params[...,9,0,:]*t + (t - 1)*(3*params[...,0,0,:]*(s - 1)**2 - 3*params[...,1,0,:]*(s - 1)**2 + 3*params[...,2,0,:]*s**2 - 3*params[...,3,0,:]*s**2 + 6*params[...,2,0,:]*s*(s - 1) - 3*params[...,1,0,:]*s*(2*s - 2)) - params[...,0,0,:]*(t - 1) + params[...,3,0,:]*(t - 1) + params[...,6,0,:]*t**3 - params[...,9,0,:]*t**3 + 3*params[...,4,0,:]*t*(t - 1)**2 - 3*params[...,5,0,:]*t**2*(t - 1) + 3*params[...,10,0,:]*t**2*(t - 1) - 3*params[...,11,0,:]*t*(t - 1)**2)**2 + (params[...,0,1,:]*(t - 1)**3 - params[...,3,1,:]*(t - 1)**3 + t*(3*params[...,8,1,:]*(s - 1)**2 - 3*params[...,9,1,:]*(s - 1)**2 + 3*params[...,6,1,:]*s**2 - 3*params[...,7,1,:]*s**2 - 6*params[...,7,1,:]*s*(s - 1) + 3*params[...,8,1,:]*s*(2*s - 2)) - params[...,6,1,:]*t + params[...,9,1,:]*t + (t - 1)*(3*params[...,0,1,:]*(s - 1)**2 - 3*params[...,1,1,:]*(s - 1)**2 + 3*params[...,2,1,:]*s**2 - 3*params[...,3,1,:]*s**2 + 6*params[...,2,1,:]*s*(s - 1) - 3*params[...,1,1,:]*s*(2*s - 2)) - params[...,0,1,:]*(t - 1) + params[...,3,1,:]*(t - 1) + params[...,6,1,:]*t**3 - params[...,9,1,:]*t**3 + 3*params[...,4,1,:]*t*(t - 1)**2 - 3*params[...,5,1,:]*t**2*(t - 1) + 3*params[...,10,1,:]*t**2*(t - 1) - 3*params[...,11,1,:]*t*(t - 1)**2)**2 + (params[...,0,2,:]*(t - 1)**3 - params[...,3,2,:]*(t - 1)**3 + t*(3*params[...,8,2,:]*(s - 1)**2 - 3*params[...,9,2,:]*(s - 1)**2 + 3*params[...,6,2,:]*s**2 - 3*params[...,7,2,:]*s**2 - 6*params[...,7,2,:]*s*(s - 1) + 3*params[...,8,2,:]*s*(2*s - 2)) - params[...,6,2,:]*t + params[...,9,2,:]*t + (t - 1)*(3*params[...,0,2,:]*(s - 1)**2 - 3*params[...,1,2,:]*(s - 1)**2 + 3*params[...,2,2,:]*s**2 - 3*params[...,3,2,:]*s**2 + 6*params[...,2,2,:]*s*(s - 1) - 3*params[...,1,2,:]*s*(2*s - 2)) - params[...,0,2,:]*(t - 1) + params[...,3,2,:]*(t - 1) + params[...,6,2,:]*t**3 - params[...,9,2,:]*t**3 + 3*params[...,4,2,:]*t*(t - 1)**2 - 3*params[...,5,2,:]*t**2*(t - 1) + 3*params[...,10,2,:]*t**2*(t - 1) - 3*params[...,11,2,:]*t*(t - 1)**2)**2) - ((params[...,0,0,:]*(s - 1)**3 - params[...,9,0,:]*(s - 1)**3 - s*(3*params[...,3,0,:]*(t - 1)**2 - 3*params[...,4,0,:]*(t - 1)**2 + 3*params[...,5,0,:]*t**2 - 3*params[...,6,0,:]*t**2 + 6*params[...,5,0,:]*t*(t - 1) - 3*params[...,4,0,:]*t*(2*t - 2)) + params[...,3,0,:]*s - params[...,6,0,:]*s + (s - 1)*(3*params[...,0,0,:]*(t - 1)**2 - 3*params[...,11,0,:]*(t - 1)**2 - 3*params[...,9,0,:]*t**2 + 3*params[...,10,0,:]*t**2 + 6*params[...,10,0,:]*t*(t - 1) - 3*params[...,11,0,:]*t*(2*t - 2)) - params[...,0,0,:]*(s - 1) + params[...,9,0,:]*(s - 1) - params[...,3,0,:]*s**3 + params[...,6,0,:]*s**3 - 3*params[...,1,0,:]*s*(s - 1)**2 + 3*params[...,2,0,:]*s**2*(s - 1) - 3*params[...,7,0,:]*s**2*(s - 1) + 3*params[...,8,0,:]*s*(s - 1)**2)*(params[...,0,0,:]*(t - 1)**3 - params[...,3,0,:]*(t - 1)**3 + t*(3*params[...,8,0,:]*(s - 1)**2 - 3*params[...,9,0,:]*(s - 1)**2 + 3*params[...,6,0,:]*s**2 - 3*params[...,7,0,:]*s**2 - 6*params[...,7,0,:]*s*(s - 1) + 3*params[...,8,0,:]*s*(2*s - 2)) - params[...,6,0,:]*t + params[...,9,0,:]*t + (t - 1)*(3*params[...,0,0,:]*(s - 1)**2 - 3*params[...,1,0,:]*(s - 1)**2 + 3*params[...,2,0,:]*s**2 - 3*params[...,3,0,:]*s**2 + 6*params[...,2,0,:]*s*(s - 1) - 3*params[...,1,0,:]*s*(2*s - 2)) - params[...,0,0,:]*(t - 1) + params[...,3,0,:]*(t - 1) + params[...,6,0,:]*t**3 - params[...,9,0,:]*t**3 + 3*params[...,4,0,:]*t*(t - 1)**2 - 3*params[...,5,0,:]*t**2*(t - 1) + 3*params[...,10,0,:]*t**2*(t - 1) - 3*params[...,11,0,:]*t*(t - 1)**2) + (params[...,0,1,:]*(s - 1)**3 - params[...,9,1,:]*(s - 1)**3 - s*(3*params[...,3,1,:]*(t - 1)**2 - 3*params[...,4,1,:]*(t - 1)**2 + 3*params[...,5,1,:]*t**2 - 3*params[...,6,1,:]*t**2 + 6*params[...,5,1,:]*t*(t - 1) - 3*params[...,4,1,:]*t*(2*t - 2)) + params[...,3,1,:]*s - params[...,6,1,:]*s + (s - 1)*(3*params[...,0,1,:]*(t - 1)**2 - 3*params[...,11,1,:]*(t - 1)**2 - 3*params[...,9,1,:]*t**2 + 3*params[...,10,1,:]*t**2 + 6*params[...,10,1,:]*t*(t - 1) - 3*params[...,11,1,:]*t*(2*t - 2)) - params[...,0,1,:]*(s - 1) + params[...,9,1,:]*(s - 1) - params[...,3,1,:]*s**3 + params[...,6,1,:]*s**3 - 3*params[...,1,1,:]*s*(s - 1)**2 + 3*params[...,2,1,:]*s**2*(s - 1) - 3*params[...,7,1,:]*s**2*(s - 1) + 3*params[...,8,1,:]*s*(s - 1)**2)*(params[...,0,1,:]*(t - 1)**3 - params[...,3,1,:]*(t - 1)**3 + t*(3*params[...,8,1,:]*(s - 1)**2 - 3*params[...,9,1,:]*(s - 1)**2 + 3*params[...,6,1,:]*s**2 - 3*params[...,7,1,:]*s**2 - 6*params[...,7,1,:]*s*(s - 1) + 3*params[...,8,1,:]*s*(2*s - 2)) - params[...,6,1,:]*t + params[...,9,1,:]*t + (t - 1)*(3*params[...,0,1,:]*(s - 1)**2 - 3*params[...,1,1,:]*(s - 1)**2 + 3*params[...,2,1,:]*s**2 - 3*params[...,3,1,:]*s**2 + 6*params[...,2,1,:]*s*(s - 1) - 3*params[...,1,1,:]*s*(2*s - 2)) - params[...,0,1,:]*(t - 1) + params[...,3,1,:]*(t - 1) + params[...,6,1,:]*t**3 - params[...,9,1,:]*t**3 + 3*params[...,4,1,:]*t*(t - 1)**2 - 3*params[...,5,1,:]*t**2*(t - 1) + 3*params[...,10,1,:]*t**2*(t - 1) - 3*params[...,11,1,:]*t*(t - 1)**2) + (params[...,0,2,:]*(s - 1)**3 - params[...,9,2,:]*(s - 1)**3 - s*(3*params[...,3,2,:]*(t - 1)**2 - 3*params[...,4,2,:]*(t - 1)**2 + 3*params[...,5,2,:]*t**2 - 3*params[...,6,2,:]*t**2 + 6*params[...,5,2,:]*t*(t - 1) - 3*params[...,4,2,:]*t*(2*t - 2)) + params[...,3,2,:]*s - params[...,6,2,:]*s + (s - 1)*(3*params[...,0,2,:]*(t - 1)**2 - 3*params[...,11,2,:]*(t - 1)**2 - 3*params[...,9,2,:]*t**2 + 3*params[...,10,2,:]*t**2 + 6*params[...,10,2,:]*t*(t - 1) - 3*params[...,11,2,:]*t*(2*t - 2)) - params[...,0,2,:]*(s - 1) + params[...,9,2,:]*(s - 1) - params[...,3,2,:]*s**3 + params[...,6,2,:]*s**3 - 3*params[...,1,2,:]*s*(s - 1)**2 + 3*params[...,2,2,:]*s**2*(s - 1) - 3*params[...,7,2,:]*s**2*(s - 1) + 3*params[...,8,2,:]*s*(s - 1)**2)*(params[...,0,2,:]*(t - 1)**3 - params[...,3,2,:]*(t - 1)**3 + t*(3*params[...,8,2,:]*(s - 1)**2 - 3*params[...,9,2,:]*(s - 1)**2 + 3*params[...,6,2,:]*s**2 - 3*params[...,7,2,:]*s**2 - 6*params[...,7,2,:]*s*(s - 1) + 3*params[...,8,2,:]*s*(2*s - 2)) - params[...,6,2,:]*t + params[...,9,2,:]*t + (t - 1)*(3*params[...,0,2,:]*(s - 1)**2 - 3*params[...,1,2,:]*(s - 1)**2 + 3*params[...,2,2,:]*s**2 - 3*params[...,3,2,:]*s**2 + 6*params[...,2,2,:]*s*(s - 1) - 3*params[...,1,2,:]*s*(2*s - 2)) - params[...,0,2,:]*(t - 1) + params[...,3,2,:]*(t - 1) + params[...,6,2,:]*t**3 - params[...,9,2,:]*t**3 + 3*params[...,4,2,:]*t*(t - 1)**2 - 3*params[...,5,2,:]*t**2*(t - 1) + 3*params[...,10,2,:]*t**2*(t - 1) - 3*params[...,11,2,:]*t*(t - 1)**2))**2
    det = det.clamp_min_(1e-10).sqrt_()
    
    return det

def coons_mtds_(s, t, patches, control_point_num = 4):
    # s,t (batch, face, 144, 1)
    if control_point_num == 4:
        coons_point_ds, coons_point_dt = coons_partial_derivative(s, t, patches)
    elif control_point_num == 8:
        coons_point_ds, coons_point_dt = coons_partial_derivative_8(s, t, patches)

    dx_ds = coons_point_ds[...,0:1]
    dx_dt = coons_point_dt[...,0:1]

    dy_ds = coons_point_ds[...,1:2]
    dy_dt = coons_point_dt[...,1:2]

    dz_ds = coons_point_ds[...,2:]
    dz_dt = coons_point_dt[...,2:]

    J = torch.cat([dx_ds,dx_dt,dy_ds,  dy_dt,dz_ds,dz_dt],dim=3)
    J = J.reshape(s.shape[0],s.shape[1],s.shape[2],3,2)
    J_T = torch.cat([dx_ds,dy_ds,dz_ds,  dx_dt,dy_dt,dz_dt],dim=3)
    J_T = J_T.reshape(s.shape[0],s.shape[1],s.shape[2],2,3)

    g = J_T @ J
    det = torch.det(g)
    det = det.clamp_min(1e-10).sqrt()
    return det

def process_primitive(
    params,
    vertex_idxs,
    face_idxs,
):
    """Process all junction curves to compute explicit patch control points."""
    
    vertices = params.clone()[:, :, vertex_idxs]
    patches = vertices[:, :, face_idxs]

    return vertices, patches

def process_patches(
    params,
    vertex_idxs,
    face_idxs,
    edge_data,
    junctions,
    junction_order,
    vertex_t
):
    """
    Process all junction curves to compute explicit patch control points.
    intput:params,Decoder's output 
    """
    
    vertices = params.clone()[:, vertex_idxs]
    for i in junction_order:
        print("\n running for i in junction_order \n")
        edge = junctions[i]
        t = torch.sigmoid(params[:, vertex_t[i]])
        vertex = bezier_sample(t[:, None, None], vertices[:, edge]).squeeze(1)
        vertices = vertices.clone()
        vertices[:, i] = vertex

        for a, b, c, d in edge_data[i]:
            if a not in junctions:
                a, b, c, d = d, c, b, a

            edge = junctions[a]
            t_a = torch.sigmoid(params[:, vertex_t[a]])
            v0_a, _, _, v3_a = edge
            if d == v0_a:
                t_d = torch.zeros_like(t_a)
            elif d == v3_a:
                t_d = torch.ones_like(t_a)
            else:
                v0_d, _, _, v3_d = junctions[d]
                t_d = torch.sigmoid(params[:, vertex_t[d]])
                if v0_a == v0_d and v3_a == v3_d:
                    pass
                elif v0_a == v3_d and v3_a == v0_d:
                    t_d = 1 - t_d
                else:
                    edge = junctions[d]
                    if a == v0_d:
                        t_a = torch.zeros_like(t_d)
                    elif a == v3_d:
                        t_a = torch.ones_like(t_d)

            curve = subbezier(t_a, t_d, vertices[:, edge])[:, 1:-1]
            vertices = vertices.clone()
            vertices[:, [b, c]] = curve

    patches = vertices[:, face_idxs]
    return vertices, patches

def make_patches(
    params,
    face_idx
):
    """
    Process all junction curves to compute explicit patch control points.
    intput: params- Decoder's output (batch_size, num_points(122) ,3)
    """
    
    vertices = params.clone()
    patches = vertices[:, face_idx]
    return vertices, patches

def bezier_sample(t, params):
    """Sample points from cubic Bezier curves defined by params at t values."""
    A = params.new_tensor([[1, 0, 0, 0],
                           [-3, 3, 0, 0],
                           [3, -6, 3, 0],
                           [-1, 3, -3, 1]])

    t = t.pow(t.new_tensor([0, 1, 2, 3]))  # [n_samples, 4]

    points = t @ A @ params  # [..., n_samples, 3]
    return points

def bezier_sample_8(t, params):
    """Sample points from cubic Bezier curves defined by params at t values."""
    A = params.new_tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                            [-7, 7, 0, 0, 0, 0, 0, 0],
                            [21, -42, 21, 0, 0, 0, 0, 0],
                            [-35, 105, -105, 35, 0, 0, 0, 0],
                            [35, -140, 210, -140, 35, 0, 0, 0],
                            [-21, 105, -210, 210, -105, 21, 0, 0],
                            [7, -42, 105, -140, 105, -42, 7, 0],
                            [-1, 7, -21, 35, -35, 21, -7, 1]
                            ])

    t = t.pow(t.new_tensor([0, 1, 2, 3, 4, 5, 6, 7]))  # [n_samples, 4]

    points = t @ A @ params  # [..., n_samples, 3]
    return points

def subbezier(t1, t2, params):
    """Compute control points for cubic Bezier curve between t1 and t2.

    t1 -- [batch_size]
    t2 -- [batch_size]
    params -- [batch_size, 4, 3]
    """
    def dB_dt(t):
        return params[:, 0]*(-3*(1-t)**2) + params[:, 1]*(3*(1-4*t+3*t**2)) \
            + params[:, 2]*(3*(2*t-3*t**2)) + params[:, 3]*(3*t**2)

    t1 = t1[:, None]
    t2 = t2[:, None]
    sub_pts = torch.empty_like(params)
    sub_pts[:, 0] = bezier_sample(t1[:, :, None], params).squeeze(1)
    sub_pts[:, 3] = bezier_sample(t2[:, :, None], params).squeeze(1)
    sub_pts[:, 1] = (t2-t1)*dB_dt(t1)/3 + sub_pts[:, 0]
    sub_pts[:, 2] = sub_pts[:, 3] - (t2-t1)*dB_dt(t2)/3
    
    return sub_pts

def write_curve(file, patches, res=300):
    """Write Bezier curves to an obj file."""
    
    x_linspace = torch.linspace(0, 1, res).to(patches)
    y_linspace = torch.linspace(0, 0, res).to(patches)
    verts1 = coons_points(x_linspace.flatten(),
                         y_linspace.flatten(), patches).cpu().numpy()

    x_linspace = torch.linspace(0, 1, res).to(patches)
    y_linspace = torch.linspace(1, 1, res).to(patches)
    verts2 = coons_points(x_linspace.flatten(),
                         y_linspace.flatten(), patches).cpu().numpy()

    x_linspace = torch.linspace(0, 0, res).to(patches)
    y_linspace = torch.linspace(0, 1, res).to(patches)
    verts3 = coons_points(x_linspace.flatten(),
                         y_linspace.flatten(), patches).cpu().numpy()

    x_linspace = torch.linspace(1, 1, res).to(patches)
    y_linspace = torch.linspace(0, 1, res).to(patches)
    verts4 = coons_points(x_linspace.flatten(),
                         y_linspace.flatten(), patches).cpu().numpy()

    verts = np.concatenate((verts1, verts2, verts3, verts4), axis = 1)
    n_verts = verts.shape[-2]
    # verts = verts.reshape(-1, verts.shape[-2], verts.shape[-1])

    # r, g, b = 0.5, 0.5, 0.5
    c = [0.6, 0.6, 0.6]

    with open(file, 'w') as f:
        for p, vertice in enumerate(verts):
            for x, y, z in vertice:
                f.write(f'v {x} {y} {z} {c[0]} {c[1]} {c[2]}\n')

def write_obj(file, patches, control_point_num = 4, res=30):
    """Write Coons patches to an obj file."""
    
    linspace = torch.linspace(0, 1, res).to(patches)
    s_grid, t_grid = torch.meshgrid(linspace, linspace)
    if control_point_num == 4:
        verts = coons_points(s_grid.flatten(),
                            t_grid.flatten(), patches).cpu().numpy()
    elif control_point_num == 8:
        verts = coons_points_8(s_grid.flatten(),
                            t_grid.flatten(), patches).cpu().numpy()
    n_verts = verts.shape[-2]
    # verts = verts.reshape(-1, verts.shape[-2], verts.shape[-1])
    
    colors = np.random.rand(10000, 3)
    with open(file, 'w') as f:
        for p, patch in enumerate(verts):
            c = colors[p]
            for x, y, z in patch:
                f.write(f'v {x} {y} {z} {c[0]} {c[1]} {c[2]}\n')
            for i in range(res-1):
                for j in range(res-1):
                    f.write(
                        f'f {i*res + j+2 + p*n_verts} {i*res + j+1 + p*n_verts} {(i+1)*res + j+1 + p*n_verts}\n')
                    f.write(
                        f'f {(i+1)*res + j+2 + p*n_verts} {i*res + j+2 + p*n_verts} {(i+1)*res + j+1 + p*n_verts}\n')
                    
def sample_patches(sample_num:int, patches:torch.Tensor):
    '''
    template patches -> point cloud. \n
    Input:  
        patches (B, face_num, cp_num, 3)    
    Output:  
        patches_sample_points (B,N,3)  
    '''
    batch_size = patches.shape[0]
    st = torch.empty(batch_size, patches.shape[1], sample_num**2, 2).uniform_().to(patches) # [B, patch_num, sample_num, 2]
    patches_sample_points = coons_points(st[..., 0], st[..., 1], patches).to(patches) # [B, patch_num, sample_num, 3]
    patches_sample_points = rearrange(patches_sample_points, 'B P N C->B (P N) C') # (B,N,3)
    return patches_sample_points
    
