import utils
from checkpoints import sorted_ckpts
from tools.vis_utils import render_full
from models.cam_params import CamParams
from models.frameworks import create_model
from geometry import c2w_track_spiral, poses_avg

import os
import torch
import imageio
import numpy as np
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import colorsys
import datetime
import matplotlib.pyplot as plt
from pykeops.torch import LazyTensor

from tools.plot_camera_pose import draw_camera

vertices = np.array([
-0.57735,  -0.57735,  0.57735,
0.934172,  0.356822,  0,
0.934172,  -0.356822,  0,
-0.934172,  0.356822,  0,
-0.934172,  -0.356822,  0,
0,  0.934172,  0.356822,
0,  -0.934172,  0.356822,
0.356822,  0,  0.934172,
-0.356822,  0,  0.934172,
0.57735,  0.57735,  0.57735,
-0.57735,  0.57735,  0.57735,
0.57735,  -0.57735,  0.57735,
]).reshape((-1,3), order="C")+0.5


def gen_raysdir(x,y,z):
    inputsX=np.linspace(x-1/256,x+1/256,128)
    inputsY=np.linspace(y-1/256,y+1/256,128)
    inputsZ=np.linspace(z-1/256,z+1/256,128)
    return torch.tensor(np.array(np.meshgrid(inputsX,inputsY,inputsZ)).T.reshape(4,-1,3)).float()


def gen_rays(x,y,z):
    # -0.1, 0.4  -0.23 0.38
    # -0.5, 0    -0.25 -0.04
    # 0.2, 0.5   -0.09 0.4
    inputsX=np.linspace(x[0],x[1],128)+np.random.rand(128)/(128.0/(x[1]-x[0]))
    inputsY=np.linspace(y[0],y[1],128)+np.random.rand(128)/(128.0/(y[1]-y[0]))
    inputsZ=np.linspace(z[0],z[1],128)+np.random.rand(128)/(128.0/(z[1]-z[0]))
    return torch.tensor(np.array(np.meshgrid(inputsX,inputsY,inputsZ)).T.reshape(4,-1,3)).float()

def KMeans(x, K=5, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    return cl, c

def gen_views(view):
    return torch.tensor(np.tile(view/np.linalg.norm(view),128**3).reshape(4,-1,3)).float()

def pointsadd(model, rays2, points,device,green=[90.0,140.0,0.7,1.0,0.6,1.0],minsig=10):
    for j in vertices:
        raw_ret_i_j = model.get_coarse_fn()(
                    inputs=rays2,
                    viewdirs=gen_views(j).to(device),
                    batched_info=None,
                    detailed_output=False
            )
        hsvs = utils.rgb2hsv(raw_ret_i_j['rgb']) # TOMER
        #minsig=torch.topk(sigmas.flatten(),2000)[0][-1].item()
        h_vals_low = torch.index_select(hsvs,2, torch.tensor([0]).to(device).long()).reshape(4,-1) > green[0]
        h_vals_high = torch.index_select(hsvs,2, torch.tensor([0]).to(device).long()).reshape(4,-1) < green[1]
        v_vals_low = torch.index_select(hsvs,2, torch.tensor([1]).to(device).long()).reshape(4,-1) > green[2]
        v_vals_high = torch.index_select(hsvs,2, torch.tensor([1]).to(device).long()).reshape(4,-1) < green[3]
        s_vals_low = torch.index_select(hsvs,2, torch.tensor([2]).to(device).long()).reshape(4,-1) > green[4]
        s_vals_high = torch.index_select(hsvs,2, torch.tensor([2]).to(device).long()).reshape(4,-1) < green[5]
        sigmas=torch.nn.functional.relu(raw_ret_i_j['sigma']) #.cpu().detach().numpy().reshape(-1) #(mask on GPU)
        sigmas= sigmas>minsig
        green_indices = (h_vals_low & 
                        h_vals_high & 
                        v_vals_low & 
                        s_vals_low & 
                        v_vals_high & 
                        s_vals_high & 
                        sigmas)
        points = torch.cat((points, rays2[green_indices]))
        points=torch.unique(points,dim=0)
    return points

def small_cubes(x,y,z):
    ret = [
    [[-1,0], [-1,0], [0,0.5]],
    [[-1,0], [0,1], [0,0.5]],
    [[-1,0], [-1,0], [0.5,1]],
    [[-1,0], [0,1], [0.5,1]],
    [[0,1], [-1,0], [0,0.5]],
    [[0,1], [0,1], [0,0.5]],
    [[0,1], [-1,0], [0.5,1]],
    [[0,1], [0,1], [0.5,1]],
    ]
    # inputsX=np.linspace(x[0],x[1],3)
    # inputsY=np.linspace(y[0],y[1],3)
    # inputsZ=np.linspace(z[0],z[1],3)
    # return np.array(np.meshgrid(inputsX,inputsY,inputsZ)).T.reshape(4,-1,3)
    return ret

def epic_render(args):
    #--------------
    # parameters
    #--------------  
    utils.cond_mkdir('out')

    #--------------
    # Load model
    #--------------
    device_ids = args.device_ids
    device = "cuda:{}".format(device_ids[0])
    exp_dir = args.training.exp_dir
    print("=> Experiments dir: {}".format(exp_dir))

    model, render_kwargs_train, render_kwargs_test, grad_vars = create_model(
        args, model_type=args.model.framework)

    if args.training.ckpt_file is None or args.training.ckpt_file == 'None':
        # automatically load 'final_xxx.pt' or 'latest.pt'
        ckpt_file = sorted_ckpts(os.path.join(exp_dir, 'ckpts'))[-1]
    else:
        ckpt_file = args.training.ckpt_file

    print("=> Loading ckpt file: {}".format(ckpt_file))
    state_dict = torch.load(ckpt_file, map_location=device)
    model_dict = state_dict['model']
    model = model.to(device)
    model.load_state_dict(model_dict)
    
    #--------------
    # Load camera parameters
    #--------------
    cam_params = CamParams.from_state_dict(state_dict['cam_param'])
    H = cam_params.H0
    W = cam_params.W0
    c2ws = cam_params.get_camera2worlds().data.cpu().numpy()
    intr = cam_params.get_intrinsic(H, W).data.cpu().numpy()

    near = args.data.near
    far = args.data.far
    c2w_center = poses_avg(c2ws)
    up = c2ws[:, :3, 1].sum(0)
    rads = np.percentile(np.abs(c2ws[:, :3, 3]), 80, 0)
    focus_distance = (far - near) * 0.7 + near
    extrinsics = np.linalg.inv(c2ws)

    # model.query_sigma(gen_rays())
    # print("worked!")
    #print(gen_rays().shape)
    #print(vertices[0].shape)
    cam_width = 0.0064 * 5 / 2 /2
    cam_height = 0.0048 * 5 / 2 /2
    scale_focal = 5.
    min_values, max_values = draw_camera(intr, cam_width, cam_height, scale_focal, extrinsics, False)
    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    print(mid_x - max_range, mid_x + max_range)
    print(mid_y - max_range, mid_y + max_range)
    print(mid_z - max_range, mid_z + max_range)
    # return
    with torch.no_grad():
        
        small_cubes_sides = small_cubes([X_min,X_max],[Y_min,Y_max],[Z_min,Z_max])

        points = torch.empty((0,3)).to(device)

        for sides in small_cubes_sides:
            rays=gen_rays(sides[0], sides[1], sides[2]).to(device)
            points = pointsadd(model, rays,points,device,minsig=5)
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(points.T.cpu()[0], points.T.cpu()[1],points.T.cpu()[2] )
            plt.savefig('sigmas.png')
            #print(high_sigmas.shape)
            #print(points.shape)
            firstscan=points
            for i in range(firstscan.shape[0]):
                incord=firstscan[i].cpu().detach().numpy()
                rays2=gen_raysdir(incord[0],incord[1],incord[2]).to(device) 
                points = pointsadd(model, rays2,points,device)
                if points.shape[0]>50000000:
                    break

        torch.save(points,"green_points.pt")

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(points.T.cpu()[0], points.T.cpu()[1],points.T.cpu()[2] )
        plt.savefig('points.png')
        means=KMeans(points)

        # for i in vertices:
        #     for j in range(128**3):
        #         if(sigmas[j]>110):
        #             incord=rays[j//524288][j%524288].cpu().detach().numpy()
        #             print(incord,sigmas[j])
        #             rays2=gen_raysdir(incord[0],incord[1],incord[2])
        #             raw_ret_i = model.get_coarse_fn()(
        #                 inputs=rays2,
        #                 viewdirs=gen_views(i).to(device),
        #                 batched_info=None,
        #                 detailed_output=False
        #             )
        #             #print(raw_ret_i['rgb'].shape)
        #             #print(raw_ret_i['sigma'].shape)
        #             # First sampling - now check sigmas to do second sampling
        #             for j in range(4):
        #                 for k in range(524288):
        #                     a=raw_ret_i['rgb'][j][k].cpu()
        #                     hsv=colorsys.rgb_to_hsv(a[0],a[1],a[2])
        #                     if hsv[0]>0.25 and  hsv[0]<140/360 and hsv[1]>0.7 and hsv[2]>0.6:
        #                         print("with view",i,"at position",rays[j][k],"We get",raw_ret_i['sigma'][j][k])
        #                         c+=1
        # print(c)
parser=utils.create_args_parser()
args,unknown=parser.parse_known_args()
config = utils.load_config(args,unknown)
epic_render(config)
