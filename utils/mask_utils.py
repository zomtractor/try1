
from torchvision.transforms import ToTensor
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage import io
import numpy as np
from glob import glob
import lpips
import warnings
warnings.filterwarnings("ignore")

def compare_lpips(img1, img2, loss_fn_alex):
    to_tensor=ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)
    output_lpips = loss_fn_alex(img1_tensor.cuda(), img2_tensor.cuda())
    return output_lpips.cpu().detach().numpy()[0,0,0,0]

def compare_score(img1,img2,img_seg):
    # Return the G-PSNR, S-PSNR, Global-PSNR and Score
    # This module is for the MIPI 2023 Challange: https://codalab.lisn.upsaclay.fr/competitions/9402
    mask_type_list=['glare','streak','global']
    metric_dict={'glare':0,'streak':0,'global':0}
    for mask_type in mask_type_list:
        mask_area,img_mask=extract_mask(img_seg)[mask_type]
        if mask_area>0:
            img_gt_masked=img1*img_mask
            img_input_masked=img2*img_mask
            input_mse=compare_mse(img_gt_masked, img_input_masked)/(255*255*mask_area)
            input_psnr=10 * np.log10((1.0 ** 2) / input_mse)
            metric_dict[mask_type]=input_psnr
        else:
            metric_dict.pop(mask_type)
    return metric_dict

def extract_mask(img_seg):
    # Return a dict with 3 masks including streak,glare,global(whole image w/o light source), masks are returned in 3ch.
    # glare: [255,255,0]
    # streak: [255,0,0]
    # light source: [0,0,255]
    # others: [0,0,0]
    mask_dict={}
    streak_mask=(img_seg[:,:,0]-img_seg[:,:,1])/255
    glare_mask=(img_seg[:,:,1])/255
    global_mask=(255-img_seg[:,:,2])/255
    mask_dict['glare']=[np.sum(glare_mask)/(512*512),np.expand_dims(glare_mask,2).repeat(3,axis=2)] #area, mask
    mask_dict['streak']=[np.sum(streak_mask)/(512*512),np.expand_dims(streak_mask,2).repeat(3,axis=2)]
    mask_dict['global']=[np.sum(global_mask)/(512*512),np.expand_dims(global_mask,2).repeat(3,axis=2)]
    return mask_dict

def calculate_metrics(gt_path,input_path,mask_path,in_loss_fn_alex):
    loss_fn_alex = in_loss_fn_alex
    gt_folder = gt_path + '/*'
    input_folder = input_path + '/*'
    gt_list = sorted(glob(gt_folder))
    input_list = sorted(glob(input_folder))
    mask_folder = mask_path + '/*'
    mask_list= sorted(glob(mask_folder))
    assert len(gt_list) == len(input_list)
    n = len(gt_list)

    ssim, psnr, lpips_val = 0, 0, 0
    score_dict={'glare':0,'streak':0,'global':0,'glare_num':0,'streak_num':0,'global_num':0}
    for i in range(n):
        img_gt = io.imread(gt_list[i])
        img_input = io.imread(input_list[i])
        ssim += compare_ssim(img_gt, img_input, multichannel=True)
        psnr += compare_psnr(img_gt, img_input, data_range=255)
        lpips_val += compare_lpips(img_gt, img_input, loss_fn_alex)
        img_seg=io.imread(mask_list[i])
        metric_dict=compare_score(img_gt,img_input,img_seg)
        for key in metric_dict.keys():
            score_dict[key]+=metric_dict[key]
            score_dict[key+'_num']+=1
    ssim /= n
    psnr /= n
    lpips_val /= n
    for key in ['glare','streak','global']:
        if score_dict[key+'_num'] == 0:
            assert False, "Error, No mask in this type!"
        score_dict[key]/= score_dict[key+'_num']
    score_dict['score']=1/3*(score_dict['glare']+score_dict['global']+score_dict['streak'])
    return psnr,ssim,lpips_val,score_dict['score'],score_dict['glare'],score_dict['streak']