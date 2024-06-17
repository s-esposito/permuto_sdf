#!/usr/bin/env python3

#calculates psnr for the task of novel view synthesis. 
# It assumes we have created our images using create_my_images.py and they are at PACKAGE_ROOT/results/output_permuto_sdf_images/
#we can only evaluate the models that were trained without mask supervision because neus only provides those results

####Call with######
# ./permuto_sdf_py/experiments/evaluation/evaluate_psnr.py --dataset dtu --scene dtu_scan83


import torch
import csv
import torchvision
from PIL import Image

import sys
import os
import numpy as np
import time
import argparse

import permuto_sdf
from easypbr  import *
from dataloaders import *
from permuto_sdf  import NGPGui
from permuto_sdf_py.utils.common_utils import create_dataloader
import permuto_sdf_py.paths.list_of_training_scenes as list_scenes


from piq import psnr, ssim, LPIPS

import subprocess




config_file="train_permuto_sdf.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)





# stores the results for a certain scene
class EvalResultsPerScene:
    def __init__(self, render_mode):
        self.render_mode = render_mode
        self.imgs_results = {}

    def update(self, img_name, psnr, ssim, lpips):
        # check if img_name already in results
        if img_name in self.imgs_results:
            print(f"[bold yellow]WARNING[/bold yellow]: {img_name} already evaluated, overwriting")
            
        self.imgs_results[img_name] = {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips
        }

    def results_averaged(self):
        psnr = self.psnr_avg()
        ssim = self.ssim_avg()
        lpips = self.lpips_avg()

        return {"psnr": psnr, "ssim": ssim, "lpips": lpips}

    def psnr_avg(self):
        total_psnr = 0
        for img_name, img_res in self.imgs_results.items():
            total_psnr += img_res["psnr"]
        psnr = total_psnr / len(self.imgs_results)
        return psnr

    def ssim_avg(self):
        total_ssim = 0
        for img_name, img_res in self.imgs_results.items():
            total_ssim += img_res["ssim"]
        ssim = total_ssim / len(self.imgs_results)
        return ssim

    def lpips_avg(self):
        total_lpips = 0
        for img_name, img_res in self.imgs_results.items():
            total_lpips += img_res["lpips"]
        lpips = total_lpips / len(self.imgs_results)
        return lpips

    def save_to_csv(self, save_path):
        all_rows = []

        print(f"results saved in {save_path}")
        file_path = os.path.join(save_path, f"{self.render_mode}.csv")

        # save to csv file
        with open(file_path, "w") as csv_file:
            writer = csv.writer(csv_file)

            row = ["img_name", "psnr", "ssim", "lpips"]
            
            for img_name, img_res in self.imgs_results.items():
                row = [img_name, img_res["psnr"], img_res["ssim"], img_res["lpips"]]
                writer.writerow(row)
                all_rows.append(row)
            
            res_avg = self.results_averaged()
            row = ["avg", res_avg["psnr"], res_avg["ssim"], res_avg["lpips"]]
            writer.writerow(row)
            all_rows.append(row)

        return all_rows    


def image2numpy(pil_image, use_lower_left_origin=False):
    """Convert a PIL Image to a numpy array with values in 0 and 1."""
    if use_lower_left_origin:
        # flip vertically
        pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
    return np.array(pil_image) / 255


def image2tensor(pil_image, device="cpu"):
    """Convert a PIL Image to a torch tensor with values in 0 and 1."""
    np_array = image2numpy(pil_image)
    return torch.from_numpy(np_array).float().to(device)


# evaluates the model
@torch.no_grad()
def eval_rendered_imgs(renders_path):
    """

    out:
        list of results (EvalResultsPerScene), one for each render mode
    """
    # iterate over folders in renders_path
    # each folder contains a different render mode
    # (e.g. "volumetric", "sphere_traced", ...)

    # check if path exists
    if not os.path.exists(renders_path):
        print(f"[bold red]ERROR[/bold red]: renders path {renders_path} for evaluation does not exist")
        exit(1)
        
    # list all folders in renders_path
    render_modes = []
    for name in os.listdir(renders_path):
        if os.path.isdir(os.path.join(renders_path, name)):
            render_modes.append(name)
    print(f"found renders for rendering modalities: {render_modes}")

    render_modes_paths = [os.path.join(renders_path, folder) for folder in render_modes]
    
    results = []
    # unmasked
    for render_mode_path, render_mode in zip(render_modes_paths, render_modes):
        # print(f"evaluating render mode {render_mode}")

        # check if "gt" and "rgb" folders exists
        if (
            os.path.exists(os.path.join(render_mode_path, "gt"))
            and os.path.exists(os.path.join(render_mode_path, "rgb"))
        ):
            # 

            # get all images filenames in gt 
            # "000.png", "001.png", ... "999.png"
            img_filenames = os.listdir(os.path.join(render_mode_path, "gt"))
            # sort by name
            img_filenames.sort()

            # list all images in gt
            gt_path = os.path.join(render_mode_path, "gt")
            gt_imgs_paths = sorted(
                [os.path.join(gt_path, img_filename) for img_filename in img_filenames]
            )

            # load corresponding images in "rgb"
            rgb_path = os.path.join(render_mode_path, "rgb")
            pred_imgs_paths = sorted(
                [os.path.join(rgb_path, img_filename) for img_filename in img_filenames]
            )

            # load images and compute psnr, ssim, lpips

            test_results = EvalResultsPerScene(render_mode)
            print(f"[bold black]evaluating {render_mode}[/bold black]")
            print("[bold black]img_name, psnr, ssim, lpips[/bold black]")
            for img_filename, gt_img_path, pred_img_path in zip(img_filenames, gt_imgs_paths, pred_imgs_paths):
                
                img_name = img_filename.split(".")[0]
                
                gt_img_pil = Image.open(gt_img_path)
                pred_img_pil = Image.open(pred_img_path)
                
                gt_rgb = image2tensor(gt_img_pil).cuda()
                pred_rgb = image2tensor(pred_img_pil).cuda()
                
                gt_rgb_tensor = gt_rgb.cuda()
                pred_rgb_tensor = pred_rgb.cuda()
                gt_rgb_tensor = gt_rgb_tensor.permute(2, 0, 1).unsqueeze(0)
                pred_rgb_tensor = pred_rgb_tensor.permute(2, 0, 1).unsqueeze(0)
                
                psnr_val = psnr(pred_rgb_tensor, gt_rgb_tensor, data_range=1.0).item()
                ssim_val = ssim(pred_rgb_tensor, gt_rgb_tensor, data_range=1.0).item()
                
                # print("pred_rgb_tensor", pred_rgb_tensor.shape)
                # print("gt_rgb_tensor", gt_rgb_tensor.shape)
                # lpips_val = LPIPS()(pred_rgb_tensor, gt_rgb_tensor).item()
                lpips_val = 0.0
                
                print(f"[bold black]{img_name}[/bold black]", psnr_val, ssim_val, lpips_val)
                
                test_results.update(img_name, psnr_val, ssim_val, lpips_val)
                
            results.append(test_results)

    # masked
    for render_mode_path, render_mode in zip(render_modes_paths, render_modes):
        # print(f"evaluating render mode {render_mode}")

        # check if "masked_gt" and "masked_rgb" folders exists
        if (
            os.path.exists(os.path.join(render_mode_path, "masked_gt")) 
            and os.path.exists(os.path.join(render_mode_path, "masked_rgb")) 
        ):
            # 

            # get all images filenames in gt 
            # "000.png", "001.png", ... "999.png"
            img_filenames = os.listdir(os.path.join(render_mode_path, "masked_gt"))
            # sort by name
            img_filenames.sort()

            # list all images in gt
            gt_path = os.path.join(render_mode_path, "masked_gt")
            gt_imgs_paths = sorted(
                [os.path.join(gt_path, img_filename) for img_filename in img_filenames]
            )

            # load corresponding images in "masked_rgb"
            rgb_path = os.path.join(render_mode_path, "masked_rgb")
            pred_imgs_paths = sorted(
                [os.path.join(rgb_path, img_filename) for img_filename in img_filenames]
            )

            # load images and compute psnr, ssim, lpips

            test_results = EvalResultsPerScene(render_mode + "_masked")
            print(f"[bold black]evaluating {render_mode}[/bold black]")
            print("[bold black]img_name, psnr, ssim, lpips[/bold black]")
            for img_filename, gt_img_path, pred_img_path in zip(img_filenames, gt_imgs_paths, pred_imgs_paths):
                
                img_name = img_filename.split(".")[0]
                
                gt_img_pil = Image.open(gt_img_path)
                pred_img_pil = Image.open(pred_img_path)
                
                gt_rgb = image2tensor(gt_img_pil).cuda()
                pred_rgb = image2tensor(pred_img_pil).cuda()
                
                gt_rgb_tensor = gt_rgb.cuda()
                pred_rgb_tensor = pred_rgb.cuda()
                gt_rgb_tensor = gt_rgb_tensor.permute(2, 0, 1).unsqueeze(0)
                pred_rgb_tensor = pred_rgb_tensor.permute(2, 0, 1).unsqueeze(0)
                
                psnr_val = psnr(pred_rgb_tensor, gt_rgb_tensor, data_range=1.0).item()
                ssim_val = ssim(pred_rgb_tensor, gt_rgb_tensor, data_range=1.0).item()
                # lpips_val = LPIPS()(pred_rgb_tensor, gt_rgb_tensor).item()
                lpips_val = 0.0
                
                print(f"[bold black]{img_name}[/bold black]", psnr_val, ssim_val, lpips_val)
                
                test_results.update(img_name, psnr_val, ssim_val, lpips_val)
                
            results.append(test_results)

    return results


def run():

    #argparse
    parser = argparse.ArgumentParser(description='Quantitative comparison')
    parser.add_argument('--dataset', required=True,  default="",  help="dataset which can be dtu or bmvs")
    parser.add_argument('--scene', required=True,  default="",  help="scene name")
    args = parser.parse_args()

    #get the results path which will be at the root of the permuto_sdf package 
    permuto_sdf_root=os.path.dirname(os.path.abspath(permuto_sdf.__file__))
    # ckpts
    checkpoint_path = os.path.join(permuto_sdf_root, "checkpoints")


    #params
    dataset=args.dataset
    with_mask=True
    low_res=False

    #path of my images
    permuto_sdf_root=os.path.dirname(os.path.abspath(permuto_sdf.__file__))

    # prepare output
    eval_res = {}
    data_splits = ["train", "test"]
    for data_split in data_splits:
        eval_res[data_split] = dict()

    # run evaluation for each eval mode
    for data_split, eval_dict in eval_res.items():
        print(f"\nrunning evaluation on {data_split} set")
        
        renders_path = os.path.join(permuto_sdf_root, "checkpoints", args.scene, "200000", "renders", data_split)
        
        # evaluate
        render_modes_eval_res = eval_rendered_imgs(renders_path)
        for res in render_modes_eval_res:
            res_avg = res.results_averaged()
            eval_dict.update(res_avg)
            # print results
            print(f"render mode: {res.render_mode}")
            for key, value in res_avg.items():
                print(f"{key}: {value}")
            # store results to csv
            res.save_to_csv(renders_path)


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