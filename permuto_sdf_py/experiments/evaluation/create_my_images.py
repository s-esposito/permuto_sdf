#!/usr/bin/env python3

#This script renders images from our trained model

#For this script the checkpoints are in ./permuto_sdf_py/scripts_for_paper/list_of_checkpoints.py
#later we can use the images to compare with neus
#images from neus can be found here: https://github.com/Totoro97/NeuS/issues/34

###CALL with 
# ./permuto_sdf_py/experiments/evaluation/create_my_images.py --dataset dtu --comp_name comp_1  [--with_mask]


import torch

import sys
import os
import numpy as np
from tqdm import tqdm
import time
import torchvision
import argparse

import easypbr
from easypbr  import *
from dataloaders import *

import permuto_sdf
from permuto_sdf  import TrainParams
from permuto_sdf  import OccupancyGrid
from permuto_sdf_py.models.models import SDF
from permuto_sdf_py.models.models import RGB
from permuto_sdf_py.models.models import NerfHash
from permuto_sdf_py.models.models import Colorcal
from permuto_sdf_py.utils.common_utils import create_dataloader
from permuto_sdf_py.utils.common_utils import create_bb_for_dataset
from permuto_sdf_py.utils.common_utils import nchw2lin
from permuto_sdf_py.utils.common_utils import lin2nchw
from permuto_sdf_py.utils.permuto_sdf_utils import load_from_checkpoint
from permuto_sdf_py.train_permuto_sdf import run_net_in_chunks
from permuto_sdf_py.train_permuto_sdf import HyperParamsPermutoSDF
import permuto_sdf_py.paths.list_of_checkpoints as list_chkpts
import permuto_sdf_py.paths.list_of_training_scenes as list_scenes


config_file="train_permuto_sdf.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_grad_enabled(False)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)




def run():
    #argparse
    parser = argparse.ArgumentParser(description='prepare dtu evaluation')
    parser.add_argument('--dataset', required=True,  default="",  help="dataset which can be dtu or bmvs")
    parser.add_argument('--scene', required=True,  default="",  help="scene name")
    parser.add_argument('--comp_name', required=True,  help='Tells which computer are we using which influences the paths for finding the data')
    parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
    parser.add_argument('--train_bg', action='store_true', help="Set this to true to train bg")
    args = parser.parse_args()
    hyperparams=HyperParamsPermutoSDF()


    #get the results path which will be at the root of the permuto_sdf package 
    permuto_sdf_root=os.path.dirname(os.path.abspath(permuto_sdf.__file__))
    # ckpts
    checkpoint_path=os.path.join(permuto_sdf_root, "checkpoints")



    #####PARAMETERS#######
    with_viewer=False
    print("args.dataset", args.dataset)
    print("args.with_mask", args.with_mask)
    print("with_viewer", with_viewer)
    chunk_size=3000
    iter_nr_for_anneal=9999999
    cos_anneal_ratio=1.0
    low_res=False


    aabb = create_bb_for_dataset(args.dataset)

    #params for rendering
    model_sdf=SDF(in_channels=3, boundary_primitive=aabb, geom_feat_size_out=hyperparams.sdf_geom_feat_size, nr_iters_for_c2f=hyperparams.sdf_nr_iters_for_c2f).to("cuda")
    model_rgb=RGB(in_channels=3, boundary_primitive=aabb, geom_feat_size_in=hyperparams.sdf_geom_feat_size, nr_iters_for_c2f=hyperparams.rgb_nr_iters_for_c2f).to("cuda")
    model_bg=NerfHash(4, boundary_primitive=aabb, nr_iters_for_c2f=hyperparams.background_nr_iters_for_c2f ).to("cuda") 
    if hyperparams.use_occupancy_grid:
        occupancy_grid=OccupancyGrid(256, 1.0, [0,0,0])
    else:
        occupancy_grid=None
    model_sdf.train(False)
    model_rgb.train(False)
    model_bg.train(False)

    color_mngr=ColorMngr()

    scan_name = args.scene

    for i in range(2):
        if i==0:
            cur_mode="train"
        else:
            cur_mode="test"

        loader_train, loader_test= create_dataloader(config_path, args.dataset, scan_name, low_res, args.comp_name, True)
        if i==0:
            loader_cur=loader_train
        else:
            loader_cur=loader_test


        nr_samples=loader_cur.nr_samples()
        print("nr_samples", nr_samples)


        #get the list of checkpoints
        # config_training="with_mask_"+str(args.with_mask) 
        # scene_config=args.dataset+"_"+config_training
        # ckpts=f"permuto_sdf_{scan_name}_default/200000"  # list_chkpts.ckpts[scene_config] 
        # ckpt_for_scene=f"permuto_sdf_{scan_name}_default/200000"
        ckpt_for_scene = f"{args.scene}/200000"
        ckpt_path_full = os.path.join(checkpoint_path,ckpt_for_scene,"models")

        #load
        load_from_checkpoint(ckpt_path_full, model_sdf, model_rgb, model_bg, occupancy_grid)

        for i in range(nr_samples):
            frame=loader_cur.get_frame_at_idx(i) 
            print("render img", frame.frame_idx)

            if frame.is_shell:
                frame.load_images()

            with torch.no_grad():
                pred_rgb_img, pred_rgb_bg_img, pred_normals_img, pred_weights_sum_img = run_net_in_chunks(frame, chunk_size, args, hyperparams, model_sdf, model_rgb, model_bg, occupancy_grid, iter_nr_for_anneal, cos_anneal_ratio, hyperparams.forced_variance_finish) 
                
                pred_rgb_img=pred_rgb_img.detach().cpu()
                pred_normals_img=pred_normals_img.detach().cpu() 
                pred_weights_sum_img=pred_weights_sum_img.detach().cpu()
                # print("pred_rgb_img", pred_rgb_img.shape)
                # print("pred_normals_img", pred_normals_img.shape)
                # print("pred_weights_sum_img", pred_weights_sum_img.shape)
            
                #get mask
                # mask_tensor=mat2tensor(frame.mask, True)
                # print("mask_tensor", mask_tensor.shape)
                
                #get gt
                # gt_img_tensor=mat2tensor(frame.rgb_8u, True).float()/255
                # print("gt_img_tensor", gt_img_tensor.shape)
                
                # gt_masked_img_tensor = gt_img_tensor*mask_tensor
                # pred_masked_img = pred_rgb_img*mask_tensor
            
            #combine with alpha from the weights
            # pred_rgba_img=torch.cat([pred_rgb_img,pred_weights_sum_img],1)
            # pred_img_mat=tensor2mat(pred_rgba_img).rgba2bgra().to_cv8u()
            pred_img_mat=tensor2mat(pred_rgb_img).rgb2bgr().to_cv8u()
            # pred_masked_img_mat=tensor2mat(pred_masked_img).rgb2bgr().to_cv8u()

            # gt_img_mat=tensor2mat(gt_img_tensor).rgb2bgr().to_cv8u()
            # gt_masked_img_mat=tensor2mat(gt_masked_img_tensor).rgb2bgr().to_cv8u()

            #normals
            pred_normals_img=torch.nn.functional.normalize(pred_normals_img, dim=1)
            pred_normals_img_vis=(pred_normals_img+1.0)*0.5
            pred_normals_img_vis=torch.cat([pred_normals_img_vis,pred_weights_sum_img],1) #concat alpha
            pred_normals_mat=tensor2mat(pred_normals_img_vis.detach()).rgba2bgra().to_cv8u()

            # #normals view coord 
            # cam=Camera()
            # cam.from_frame(frame)
            # tf_cam_world=cam.view_matrix_affine() 
            # tf_cam_world_t=torch.from_numpy(tf_cam_world.matrix()).cuda()
            # tf_cam_world_R=torch.from_numpy(tf_cam_world.matrix()).cuda()[0:3, 0:3]
            # pred_normals_lin=nchw2lin(pred_normals_img)
            # pred_normals_lin_0=torch.cat([pred_normals_lin, torch.zeros_like(pred_normals_lin)[:,0:1]  ],1)
            # pred_normals_viewcoords_lin=torch.matmul(tf_cam_world_R,pred_normals_lin.t()).t()
            # pred_normals_viewcoords_lin=torch.nn.functional.normalize(pred_normals_viewcoords_lin,dim=1)
            # pred_normals_viewcoords_lin_vis=(pred_normals_viewcoords_lin+1)*0.5
            # pred_normals_viewcoords_img_vis=lin2nchw(pred_normals_viewcoords_lin_vis, frame.height, frame.width)
            # pred_normals_viewcoords_img_vis=torch.cat([pred_normals_viewcoords_img_vis,pred_weights_sum_img],1) #concat alpha
            # pred_normals_viewcoords_mat=tensor2mat(pred_normals_viewcoords_img_vis.detach()).rgba2bgra().to_cv8u()

            #output path
            # out_img_path=os.path.join(permuto_sdf_root,"results/output_permuto_sdf_images",args.dataset, config_training, cur_mode, scan_name)
            out_img_path = os.path.join(checkpoint_path, ckpt_for_scene, "renders", cur_mode, "volumetric")
            
            #write images to file
            os.makedirs(  os.path.join(out_img_path,"rgb"), exist_ok=True)
            os.makedirs(  os.path.join(out_img_path,"gt"), exist_ok=True)
            # os.makedirs(  os.path.join(out_img_path,"masked_rgb"), exist_ok=True)
            # os.makedirs(  os.path.join(out_img_path,"masked_gt"), exist_ok=True)
            os.makedirs(  os.path.join(out_img_path,"normals"), exist_ok=True)
            # os.makedirs(  os.path.join(out_img_path,"normals_viewcoords"), exist_ok=True)

            # save gt
            # gt_img_mat.to_file(   os.path.join(out_img_path,"gt", str(frame.frame_idx)+".png"  )  )
            # gt_masked_img_mat.to_file(   os.path.join(out_img_path,"masked_gt", str(frame.frame_idx)+".png"  )  )
            pred_img_mat.to_file(   os.path.join(out_img_path,"rgb", str(frame.frame_idx)+".png"  )  )
            # pred_masked_img_mat.to_file(   os.path.join(out_img_path,"masked_rgb", str(frame.frame_idx)+".png"  )  )
            pred_normals_mat.to_file(   os.path.join(out_img_path,"normals", str(frame.frame_idx)+".png"  )  )
            
            # pred_normals_viewcoords_mat.to_file(   os.path.join(out_img_path,"normals_viewcoords", str(frame.frame_idx)+".png"  )  )

            frame.unload_images()
            # break



def main():
    run()



if __name__ == "__main__":
    main()  # This is what you would have, but the following is useful:

    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace

    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr

    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()')