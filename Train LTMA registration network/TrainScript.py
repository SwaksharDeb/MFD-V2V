import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import NCC, dice_coefficient, Grad
from timeit import default_timer
from datasets.datasetDec12 import DENSE

from PIL import Image
import os
from pathlib import Path
import animation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def save_tensor_as_gif(tensor, save_dir, base_filename="sequence"):
    """
    Save a 4D tensor as multiple GIF animations.
    
    Args:
        tensor (torch.Tensor): Shape (batch_size, time_frames, height, width)
        save_dir (str): Directory to save the GIF files
        base_filename (str): Base name for the output files
    """
    # Create directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure tensor is on CPU and convert to numpy
    tensor = tensor.cpu().detach()
    
    # Normalize to [0, 255] range
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.numpy().astype(np.uint8)
    
    # Save each batch as a separate GIF
    for batch_idx in range(tensor.shape[0]):
        frames = []
        for time_idx in range(tensor.shape[1]):
            # Convert numpy array to PIL Image
            img = Image.fromarray(tensor[batch_idx, time_idx])
            frames.append(img)
        
        # Save as GIF
        output_path = save_dir / f"{base_filename}_{batch_idx}.gif"
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=200,  # Duration between frames in milliseconds
            loop=0  # 0 means infinite loop
        )
        print(f"Saved GIF: {output_path}")


def save_model_state_dict(model):
    state_dict = {}
    for key in model.state_dict():
        state_dict[key] = model.state_dict()[key].clone()
    return state_dict


def trainer(args, model, snapshot_path):
    # model = Net2DResNet(args).cuda()
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.loss_type == "MSE":
        criterion = nn.MSELoss()
    elif args.loss_type == "NCC":
        criterion = NCC().loss
    regloss = Grad('l2').loss
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = DENSE(path=args.train_dense, trainlenscale=1, split="train", cur_img_size=args.img_size, resmode=args.resmode, series_len=args.series_len, basesize=args.databasesize, pathmask=args.pathmask)
    db_test = DENSE(path=args.test_dense, split="test", testlen=200, cur_img_size=args.img_size, resmode=args.resmode, series_len=args.series_len, basesize=args.databasesize, pathmask=args.pathmask)
    
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    params_regis = {'lr':args.regis_lr, 'stp':args.regis_stp, 'gma':args.regis_gamma}
    optimizer_regis = torch.optim.Adam([ {'params': model.parameters(), 'lr': params_regis['lr']}], weight_decay=args.weight_decay)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    loss_min_total = 9999999; loss_min_similarity = 9999999; loss_min_regularity = 9999999
    path_model_best_totalloss = ""
    path_model_best_similarity = ""
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    print("The length of train set is: {}".format(len(db_train)))

    for epoch_num in iterator:
        print("\n")
        model.train()
        t1 = default_timer()

        for i_batch, sampled_batch in enumerate(trainloader):
            if args.resmode == "voxelmorph":
                src, tar = sampled_batch['src'], sampled_batch['tar'] 
                # print(src.shape)  #torch.Size([32, 1, 12, 64, 64])
                b,b,w,h = src.shape
                src = src.reshape(-1,w,h).unsqueeze(1).cuda(); tar = tar.reshape(-1,w,h).unsqueeze(1).cuda()  #[10, 1, 32, 32]
                if "mask_src" in sampled_batch.keys():
                    mask_src, mask_tar = sampled_batch['mask_src'], sampled_batch['mask_tar']
                    mask_src = mask_src.reshape(-1,w,h).unsqueeze(1).cuda()
                    mask_tar = mask_tar.reshape(-1,w,h).unsqueeze(1).cuda()
                    input_mask = torch.cat((mask_src, mask_tar), dim=1)
                else:
                    input_mask = None
                input = torch.cat((src,tar),dim=1)
                Sdef, v, u_seq, Sdef_mask_series, _ = model(input, resmode = args.resmode, masks=input_mask)
                loss1 = criterion(tar,Sdef)
                loss2 = regloss(None, v)
                loss = loss1*args.svf_simi_w + args.svf_reg_w*loss2

                optimizer_regis.zero_grad()
                loss.backward()
                optimizer_regis.step()

                if Sdef_mask_series is not None:
                    dice = dice_coefficient(Sdef_mask_series, mask_tar, return_mean=True)
                else:
                    dice = 0

                iter_num = iter_num + 1
                model.writer.add_scalar('info/train_totalloss', loss.item(), iter_num)
                model.writer.add_scalar('info/train_similarity', loss1.item(), iter_num)
                model.writer.add_scalar('info/train_regularity', loss2.item(), iter_num)
                model.writer.add_scalar('info/train_dice', dice, iter_num)
                logging.info(f'MiccaiVoxelmorphTrain~~~~~~  epoch_num: {epoch_num} iter_num: {iter_num} src {sampled_batch["src"].shape} \
                              totalloss {loss.item()} similarity {loss1.item()} regularity {loss2.item()} dice {dice}')
            
            elif args.resmode == "TLRN" :  #resmode=""): #resmode="series"
                slices = sampled_batch['series']    #[2, 2, 9, 64, 64] [b, 2, 9, 64, 64] 
                b,b,c,w,h = slices.shape
                slices_series = slices.reshape(-1,c,w,h).cuda()  #[2, 2, 9, 64, 64]  ->  [4, 9, 64, 64]

                if "masks" in sampled_batch.keys():
                    masks = sampled_batch['masks']
                    masks_series = masks.reshape(-1,c,w,h).cuda()  #[2, 2, 9, 64, 64]  ->  [4, 9, 64, 64]
                else:
                    masks_series = None
                
                # output_dir = "output_gifs"
                # save_tensor_as_gif(slices_series, output_dir)

                Sdef_series, v_series, u_series, Sdef_mask_series, ui_series = model(slices_series, resmode = "TLRN", masks=masks_series)
                if i_batch %50 == 0:
                    phi = torch.stack(ui_series, dim=1).permute(0,4,1,2,3)
                    animation.create_grid_visualization(phi, 'deformation', img_size=64)

                loss_series = 0; loss1_series =0; loss2_series = 0; dice_series = 0
                for i in range(len(Sdef_series)):
                    Tloss1_series = criterion(slices_series[:, i+1: i+2, :,:], Sdef_series[i])
                    Tloss2_series = regloss(None, v_series[i])
                    Tloss_series = Tloss1_series*args.svf_simi_w + args.svf_reg_w2*Tloss2_series
                    if Sdef_mask_series is not None:
                        Sdef_mask = Sdef_mask_series[i]   #1,64,64
                        tar_mask = masks_series[:,i+1:i+2,:,:]  #1,64,64
                        dice = dice_coefficient(Sdef_mask, tar_mask, return_mean=True)
                    else:
                        dice = 0
                    loss_series += Tloss_series
                    loss1_series += Tloss1_series
                    loss2_series += Tloss2_series
                    dice_series += dice

                loss_series /= len(Sdef_series)
                loss1_series /= len(Sdef_series)
                loss2_series /= len(Sdef_series)
                dice_series /= len(Sdef_series)
                
                optimizer_regis.zero_grad()
                loss_series.backward()
                optimizer_regis.step()

                iter_num = iter_num + 1
                model.writer.add_scalar('info/train_totalloss', loss_series.item(), iter_num)
                model.writer.add_scalar('info/train_similarity', loss1_series.item(), iter_num)
                model.writer.add_scalar('info/train_regularity', loss2_series.item(), iter_num)
                model.writer.add_scalar('info/train_dice', dice_series, iter_num)
                logging.info(f'MiccaiTLRNTrain~~~~~~  epoch_num: {epoch_num} iter_num: {iter_num} series {slices_series.shape} totalloss {loss_series.item()} similarity {loss1_series.item()} regularity {loss2_series.item()} dice {dice_series}')
        
        test_loss = {"total_loss":[], "similarity":[], "regularity":[], "dice": []}
        with torch.no_grad():
            model.eval()
            for i_batch, sampled_batch in enumerate(testloader):
                if args.resmode == "voxelmorph": 
                    src, tar = sampled_batch['src'], sampled_batch['tar']  #[5, 1, 16, 64, 64]
                    b,b,w,h = src.shape
                    src = src.reshape(-1,w,h).unsqueeze(1).cuda(); tar = tar.reshape(-1,w,h).unsqueeze(1).cuda()  #[10, 1, 32, 32]
                    if "mask_src" in sampled_batch.keys():
                        mask_src, mask_tar = sampled_batch['mask_src'], sampled_batch['mask_tar']
                        mask_src = mask_src.reshape(-1,w,h).unsqueeze(1).cuda()
                        mask_tar = mask_tar.reshape(-1,w,h).unsqueeze(1).cuda()
                        input_mask = torch.cat((mask_src, mask_tar), dim=1)
                    else:
                        input_mask = None

                    input = torch.cat((src,tar),dim=1)
                    Sdef, v, u_seq, Sdef_mask_series, _ = model(input, resmode = args.resmode, masks=input_mask)

                    loss1 = criterion(tar,Sdef)
                    loss2 = regloss(None, v)
                    loss = loss1*args.svf_simi_w + args.svf_reg_w*loss2

                    if Sdef_mask_series is not None:
                        dice = dice_coefficient(Sdef_mask_series, mask_tar, return_mean=True)
                    else:
                        dice = 0

                    test_loss['total_loss'].append(loss.item())
                    test_loss['similarity'].append(loss1.item())
                    test_loss['regularity'].append(loss2.item())
                    test_loss['dice'].append(dice)

                elif args.resmode == "TLRN" :  #resmode=""): #resmode="series"
                    ###  part1: series
                    slices = sampled_batch['series']    #[2, 2, 9, 64, 64] [b, 2, 9, 64, 64] 
                    b,b,c,w,h = slices.shape
                    slices_series = slices.reshape(-1,c,w,h).cuda()  #[2, 2, 9, 64, 64]  ->  [4, 9, 64, 64]
                    if "masks" in sampled_batch.keys():
                        masks = sampled_batch['masks']
                        masks_series = masks.reshape(-1,c,w,h).cuda()  #[2, 2, 9, 64, 64]  ->  [4, 9, 64, 64]
                    else:
                        masks_series = None
                    
                    Sdef_series, v_series, u_series, Sdef_mask_series, _ = model(slices_series, resmode = "TLRN", masks=masks_series)

                    loss_series = 0; loss1_series =0; loss2_series = 0; dice_series=0
                    for i in range(len(Sdef_series)):
                        Tloss1_series = criterion(slices_series[:, i+1: i+2, :,:], Sdef_series[i])
                        Tloss2_series = regloss(None, v_series[i])
                        Tloss_series = Tloss1_series*args.svf_simi_w + args.svf_reg_w2*Tloss2_series
                        if Sdef_mask_series is not None:
                            Sdef_mask = Sdef_mask_series[i]   #1,64,64
                            tar_mask = masks_series[:,i+1:i+2,:,:]  #1,64,64
                            dice = dice_coefficient(Sdef_mask, tar_mask, return_mean=True)
                        else:
                            dice = 0
                        loss_series += Tloss_series
                        loss1_series += Tloss1_series
                        loss2_series += Tloss2_series
                        dice_series += dice

                    loss_series /= len(Sdef_series)
                    loss1_series /= len(Sdef_series)
                    loss2_series /= len(Sdef_series)
                    dice_series /= len(Sdef_series)

                    test_loss['total_loss'].append(loss_series.item())
                    test_loss['similarity'].append(loss1_series.item())
                    test_loss['regularity'].append(loss2_series.item())
                    test_loss['dice'].append(dice_series)

            model.writer.add_scalar('info/test_totalloss', np.mean(test_loss['total_loss']), iter_num)
            model.writer.add_scalar('info/test_similarity', np.mean(test_loss['similarity']), iter_num)
            model.writer.add_scalar('info/test_regularity', np.mean(test_loss['regularity']), iter_num)
            model.writer.add_scalar('info/test_dice', np.mean(test_loss['dice']), iter_num)

            t2 = default_timer()
            logging.info(f'\n\nMiccaiTest~~~one-ep-time: {t2-t1} epoch_num: {epoch_num}~~~  epoch_num: {epoch_num} iter_num: {iter_num} \
                         totalloss {np.mean(test_loss["total_loss"])} similarity {np.mean(test_loss["similarity"])} regularity {np.mean(test_loss["regularity"])} dice {np.mean(test_loss["dice"])}\n\n')
        
        if((epoch_num+1)%1)==0:
            try:
                os.remove(path_model)
            except:
                pass
            path_model = args.snapshot_path+ "/{}.pth".format(epoch_num)
            torch.save({
                'register': model.state_dict(),
            }, path_model)
        
        lowest_loss_test1 = np.mean(test_loss['total_loss']) < loss_min_total
        if lowest_loss_test1:
            loss_min_total = np.mean(test_loss['total_loss'])
            path_model_best_totalloss = save_model(model, args.snapshot_path, epoch_num, path_model_best_totalloss)
        
        lowest_loss_test2 = np.mean(test_loss['similarity']) < loss_min_similarity
        if lowest_loss_test2:
            loss_min_similarity = np.mean(test_loss['similarity'])
            path_model_best_similarity = save_model(model, args.snapshot_path, epoch_num, path_model_best_similarity)

def save_model(model, snapshot_path, epoch_num, path_model):
    try :
        os.remove(path_model)
    except:
        pass
    path_model = snapshot_path+ "/registration_{}.pth".format(epoch_num)
    torch.save({'registration': model.state_dict()}, path_model)
    return path_model
