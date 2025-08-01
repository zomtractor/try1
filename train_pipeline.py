import os
import torch
import cv2
from skimage import img_as_ubyte
from focal_frequency_loss import FocalFrequencyLoss as FFL
import yaml

from model import UBlock
from utils import network_parameters
import torch.optim as optim
import time
import utils
import numpy as np
import random
import math
from DataPro import get_training_data, get_validation_data
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.utils import ASLloss,ColorLoss,Blur,L1_Charbonnier_loss,img_pad,SSIM_loss,VGGLoss
from utils.mask_utils import calculate_metrics
import lpips
import warnings
from lightning.fabric import Fabric


def init_torch_config():
    warnings.filterwarnings("ignore")
    # torch.set_float32_matmul_precision('high')
    ## Set Seeds
    my_seed = 1234
    torch.backends.cudnn.benchmark = True
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    torch.set_float32_matmul_precision('high')
    fabric=Fabric(accelerator="cuda",devices=2,strategy="ddp_find_unused_parameters_true")
    fabric.launch()
    return fabric

def get_data_loaders(config, fabric):
    Train = config['TRAINING']
    OPT = config['TRAINOPTIM']
    ## DataLoaders
    print('==> Loading datasets')
    utils.mkdir(Train['VAL']['REAL_SAVE'])
    utils.mkdir(Train['VAL']['SYN_SAVE'])

    train_dataset = get_training_data(Train['TRAIN_DIR'], {'patch_size': Train['TRAIN_PS']})
    train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                              shuffle=True, num_workers=OPT['BATCH'], drop_last=True)
    real_val_dataset = get_validation_data(Train['VAL']['REAL_DIR'], {'patch_size': Train['VAL_PS']})
    real_val_loader = DataLoader(dataset=real_val_dataset, batch_size=1, shuffle=False, num_workers=2,
                            drop_last=True)
    syn_val_dataset = get_validation_data(Train['VAL']['SYN_DIR'], {'patch_size': Train['VAL_PS']})
    syn_val_loader = DataLoader(dataset=syn_val_dataset, batch_size=1, shuffle=False, num_workers=2,
                            drop_last=True)
    train_loader = fabric.setup_dataloaders(train_loader)
    real_val_loader = fabric.setup_dataloaders(real_val_loader)
    syn_val_loader = fabric.setup_dataloaders(syn_val_loader)
    return train_loader, real_val_loader, syn_val_loader

def get_loss_functions():
    ## DataLoaders
    SSIMloss = SSIM_loss().cuda()
    Charloss = L1_Charbonnier_loss().cuda()
    Vgg_loss = VGGLoss().cuda()
    Freq_loss = FFL(loss_weight=0.1, alpha=1.0).cuda()
    return SSIMloss, Charloss, Vgg_loss, Freq_loss

def load_model(config, fabric):

    Train = config['TRAINING']
    OPT = config['TRAINOPTIM']

    print('==> Build the model')
    ## Training model path direction
    mode = OPT['MODEL']['MODE']
    model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
    utils.mkdir(model_dir)
    model_restored = UBlock(base_channels=10)
    p_number = network_parameters(model_restored)
    ## Optimizer
    start_epoch = 1
    new_lr = float(OPT['LR_INITIAL'])
    optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
    model_restored, optimizer = fabric.setup(model_restored, optimizer)
    ## Scheduler (Strategy)
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                            eta_min=float(OPT['LR_MIN']))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
    checkpoint = None
    ## Resume (Continue training by a pretrained model)
    if Train['RESUME']:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        checkpoint = utils.load_checkpoint(model_restored, path_chk_rest)
        if (checkpoint is not None):
            # start_epoch = utils.load_start_epoch(path_chk_rest) + 1
            start_epoch = checkpoint['epoch'] + 1
            # utils.load_optim(optimizer, path_chk_rest)
            optimizer.load_state_dict(checkpoint['optimizer'])

            for i in range(1, start_epoch):
                scheduler.step()
            new_lr = scheduler.get_lr()[0]
            print('------------------------------------------------------------------')
            print("==> Resuming Training with learning rate:", new_lr)
            print('------------------------------------------------------------------')
        else:
            print('No checkpoint found, starting from scratch.')
    # Show the training configuration
    print(f'''==> Training details:
        ------------------------------------------------------------------
            Restoration mode:   {mode}
            Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
            Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
            Model parameters:   \\{p_number}
            Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'] + 1)}
            Batch sizes:        {OPT['BATCH']}
            Learning rate:      {OPT['LR_INITIAL']}''')
    print('------------------------------------------------------------------')
    return model_restored,checkpoint, optimizer,scheduler,start_epoch, model_dir

def load_config():
    ## Load yaml configuration file
    opt = None
    with open('config.yaml', 'r') as config:
        opt = yaml.safe_load(config)

    Train = opt['TRAINING']
    OPT = opt['TRAINOPTIM']
    mode = OPT['MODEL']['MODE']

    ## Log
    log_dir = os.path.join(Train['SAVE_DIR'], mode, 'train_logs')
    utils.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')
    return opt, writer

if __name__ == '__main__':


    def validate(model_restored, name, val_loader, best_psnr, best_ssim, best_lpips, best_score, best_Gpsnr,
                 best_Spsnr):
        model_restored.eval()
        print(f'==> Validation on {name} dataset=====================================================')
        # psnr_val_rgb = []
        # ssim_val_rgb = []
        cumulative_psnr = 0
        cumulative_ssim = 0
        cumulative_lpips = 0
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            b, c, h, w = input_.size()
            k = 16
            # pad image such that the resolution is a multiple of 32
            w_pad = (math.ceil(w / k) * k - w) // 2
            h_pad = (math.ceil(h / k) * k - h) // 2
            w_odd_pad = w_pad
            h_odd_pad = h_pad
            if w % 2 == 1:
                w_odd_pad += 1
            if h % 2 == 1:
                h_odd_pad += 1
            input_ = img_pad(input_, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
            # input_ = imageprocess.addpadding(32,input_)

            with torch.no_grad():
                restored = model_restored(input_)
                # for res, tar in zip(restored, target):
                #     psnr_val_rgb.append(utils.torchPSNR(res, tar))
                #     ssim_val_rgb.append(utils.torchSSIM(restored, target))
                if h_pad != 0:
                    restored = restored[:, :, h_pad:-h_odd_pad, :]
                if w_pad != 0:
                    restored = restored[:, :, :, w_pad:-w_odd_pad]
            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                cv2.imwrite(os.path.join(real_input_path, data_val[2][batch] + '.png'),
                            cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))

        psnr_val_rgb, ssim_val_rgb, lpips_val_rgb, score_val_rgb, Gpsnr_val_rgb, Spsnr_val_rgb = calculate_metrics(
            real_gt_path, real_input_path, real_mask_path, loss_fn_alex)

        # Save the best PSNR model of validation
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

        # Save the best SSIM model of validation
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSSIM.pth"))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

        # Save the best LPIPS model of validation
        if lpips_val_rgb < best_lpips:
            best_lpips = lpips_val_rgb
            best_epoch_lpips = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestLPIPS.pth"))
        print("[epoch %d LPIPS: %.4f --- best_epoch %d Best_LPIPS %.4f]" % (
            epoch, lpips_val_rgb, best_epoch_lpips, best_lpips))
        # Save the best score model of validation
        if score_val_rgb > best_score:
            best_score = score_val_rgb
            best_epoch_score = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestScore.pth"))
        print("[epoch %d Score: %.4f --- best_epoch %d Best_Score %.4f]" % (
            epoch, score_val_rgb, best_epoch_score, best_score))

        # Save the best Gpsnr model of validation
        if Gpsnr_val_rgb > best_Gpsnr:
            best_Gpsnr = Gpsnr_val_rgb
            best_epoch_Gpsnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestGpsnr.pth"))
        print("[epoch %d Gpsnr: %.4f --- best_epoch %d Best_Gpsnr %.4f]" % (
            epoch, Gpsnr_val_rgb, best_epoch_Gpsnr, best_Gpsnr))

        # Save the best Spsnr model of validation
        if Spsnr_val_rgb > best_Spsnr:
            best_Spsnr = Spsnr_val_rgb
            best_epoch_Spsnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSpsnr.pth"))
        print("[epoch %d Spsnr: %.4f --- best_epoch %d Best_Spsnr %.4f]" % (
            epoch, Spsnr_val_rgb, best_epoch_Spsnr, best_Spsnr))
        """ 
        # Save evey epochs of model
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        """

        writer.add_scalar(f'val/PSNR_{name}', psnr_val_rgb, epoch)
        writer.add_scalar(f'val/SSIM_{name}', ssim_val_rgb, epoch)
        writer.add_scalar(f'val/LPIPS_{name}', lpips_val_rgb, epoch)
        writer.add_scalar(f'val/Score_{name}', score_val_rgb, epoch)
        writer.add_scalar(f'val/Gpsnr_{name}', Gpsnr_val_rgb, epoch)
        writer.add_scalar(f'val/Spsnr_{name}', Spsnr_val_rgb, epoch)

    fabric =  init_torch_config()

    # Start training!
    print('==> Training start: ')
    best_psnr ,best_ssim ,best_lpips,best_GPsnr,best_Spsnr,best_score = 0,0,1000,0,0,0
    best_epoch_psnr, best_epoch_ssim,best_epoch_lpips, best_epoch_score, best_epoch_Gpsnr, best_epoch_Spsnr=0,0,0,0,0,0

    best_syn_psnr, best_syn_ssim, best_syn_lpips, best_syn_score, best_syn_Gpsnr, best_syn_Spsnr = 0, 0, 1000, 0, 0, 0
    best_syn_epoch_psnr, best_syn_epoch_ssim, best_syn_epoch_lpips, best_syn_epoch_score, best_syn_epoch_Gpsnr, best_syn_epoch_Spsnr = 0, 0, 0, 0, 0, 0

    config, writer = load_config()
    model_restored, checkpoint, optimizer,scheduler, start_epoch, model_dir = load_model(config, fabric)
    if checkpoint is not None and 'best_psnr' in checkpoint:
        best_psnr = checkpoint['best_psnr']
        best_ssim = checkpoint['best_ssim']
        best_epoch_psnr = checkpoint['best_epoch_psnr']
        best_epoch_ssim = checkpoint['best_epoch_ssim']
        best_lpips=checkpoint['best_lpips']
        best_epoch_lpips=checkpoint['best_epoch_lpips']
        best_score=checkpoint['best_score']
        best_epoch_score=checkpoint['best_epoch_score']
        best_Gpsnr=checkpoint['best_Gpsnr']
        best_epoch_Gpsnr=checkpoint['best_epoch_Gpsnr']
        best_Spsnr=checkpoint['best_Spsnr']
        best_epoch_Spsnr=checkpoint['best_epoch_Spsnr']

        best_syn_psnr = checkpoint.get('best_syn_psnr', 0)
        best_syn_ssim = checkpoint.get('best_syn_ssim', 0)
        best_syn_lpips = checkpoint.get('best_syn_lpips', 1000)
        best_syn_epoch_psnr = checkpoint.get('best_syn_epoch_psnr', 0)
        best_syn_epoch_ssim = checkpoint.get('best_syn_epoch_ssim', 0)
        best_syn_epoch_lpips = checkpoint.get('best_syn_epoch_lpips', 0)
        best_syn_score = checkpoint.get('best_syn_score', 0)
        best_syn_epoch_score = checkpoint.get('best_syn_epoch_score', 0)
        best_syn_Gpsnr = checkpoint.get('best_syn_Gpsnr', 0)
        best_syn_epoch_Gpsnr = checkpoint.get('best_syn_epoch_Gpsnr', 0)
        best_syn_Spsnr = checkpoint.get('best_syn_Spsnr', 0)
        best_syn_epoch_Spsnr = checkpoint.get('best_syn_epoch_Spsnr', 0)

        print("load indices from checkpoint succeed.")
    else:
        print('No checkpoint found, starting from scratch.')

    train_loader, real_val_loader, syn_val_loader = get_data_loaders(config, fabric)
    total_start_time = time.time()
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    # gt_path = "./dataset/Flare7Kpp/test_data/real/gt"
    # gt_path = "./dataset/Flare7Kpp/test_data/real/gt"

    Train = config['TRAINING']
    OPT = config['TRAINOPTIM']
    real_val_dir = Train['VAL']['REAL_DIR']
    real_gt_path = os.path.join(real_val_dir,'gt')
    real_input_path = Train['VAL']['REAL_SAVE']
    real_mask_path = os.path.join(real_val_dir,'mask')

    syn_val_dir = Train['VAL']['SYN_DIR']
    syn_gt_path = os.path.join(syn_val_dir,'gt')
    syn_input_path = Train['VAL']['SYN_SAVE']
    syn_mask_path = os.path.join(syn_val_dir,'mask')

    SSIMloss, Charloss, Vgg_loss, Freq_loss = get_loss_functions()

    for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_ssim_loss=0
        epoch_c1_loss=0
        epoch_vgg_loss=0
        epoch_freq_loss=0
        train_id = 1

        model_restored.train()
        for i, data in enumerate(train_loader, 0):
            # Forward propagation
            # for param in model_restored.parameters():
            #     param.grad = None
            optimizer.zero_grad()
            target = data[0].cuda()
            input_ = data[1].cuda()
            restored = model_restored(input_)

            # Compute loss
            charl1 = Charloss(restored, target)
            ssim_loss = (1 - SSIMloss(restored, target))
            # color_loss = cl(blur_rgb(restored), blur_rgb(target))
            vgg_loss=Vgg_loss(restored,target)
            freq_loss = Freq_loss(restored,target)
            loss = charl1 + ssim_loss + 1.5*freq_loss + 0.5*vgg_loss  # 损失函数
            # Back propagation
            # loss.backward()
            fabric.backward(loss)
            optimizer.step()
            epoch_ssim_loss+=ssim_loss.item()
            epoch_loss += loss.item()
            epoch_c1_loss+=charl1.item()
            epoch_vgg_loss +=vgg_loss.item()
            epoch_freq_loss+=freq_loss.item()
            if i%500 == 499:
                print(f'echo {epoch}, iter {i+1} finished.===================================================')
        ## Evaluation (Validation)
        if epoch % Train['VAL_AFTER_EVERY'] == 0:
            validate(model_restored, 'real', real_val_loader, best_psnr, best_ssim, best_lpips, best_score)
            validate(model_restored, 'syn', syn_val_loader, best_syn_psnr, best_syn_ssim, best_syn_lpips, best_syn_score,
                     best_syn_Gpsnr, best_syn_Spsnr)

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tSSIMLoss: {:.4f}\tChar1Loss: {:.4f}\tVGGLoss: {:.4f}\tFreqLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss, epoch_ssim_loss,epoch_c1_loss,epoch_vgg_loss,epoch_freq_loss,scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        # Save the last model
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "best_epoch_psnr": best_epoch_psnr,
                    "best_epoch_ssim": best_epoch_ssim,
                    "best_lpips": best_lpips,
                    "best_epoch_lpips": best_epoch_lpips,
                    "best_score": best_score,
                    "best_epoch_score": best_epoch_score,
                    "best_Gpsnr": best_Gpsnr,
                    "best_epoch_Gpsnr": best_epoch_Gpsnr,
                    "best_Spsnr": best_Spsnr,
                    "best_epoch_Spsnr": best_epoch_Spsnr,

                    "best_syn_psnr": best_syn_psnr,
                    "best_syn_ssim": best_syn_ssim,
                    "best_syn_lpips": best_syn_lpips,
                    "best_syn_epoch_psnr": best_syn_epoch_psnr,
                    "best_syn_epoch_ssim": best_syn_epoch_ssim,
                    "best_syn_epoch_lpips": best_syn_epoch_lpips,
                    "best_syn_score": best_syn_score,
                    "best_syn_epoch_score": best_syn_epoch_score,
                    "best_syn_Gpsnr": best_syn_Gpsnr,
                    "best_syn_epoch_Gpsnr": best_syn_epoch_Gpsnr,
                    "best_syn_Spsnr": best_syn_Spsnr,
                    "best_syn_epoch_Spsnr": best_syn_epoch_Spsnr


                    }, os.path.join(model_dir, "model_latest.pth"))

        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/ssim_loss', epoch_ssim_loss, epoch)
        writer.add_scalar('train/c1_loss',epoch_c1_loss, epoch)
        writer.add_scalar('train/vgg_loss', epoch_vgg_loss, epoch)
        writer.add_scalar('train/freq_loss', epoch_freq_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
    writer.close()

    total_finish_time = (time.time() - total_start_time)  # seconds
    print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))







