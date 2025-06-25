import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg16, VGG16_Weights
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import torch.backends.cudnn as cudnn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import timeit
import math
from PIL import Image
from misc import Logger, grid2fig, conf2fig
from datasets_natops import NATOPS, NATOPS_test
import sys
import random
from skimage.metrics import structural_similarity as ssim
#from DM.modules.video_flow_diffusion_model import FlowDiffusion
from modules.video_flow_diffusion_model import FlowDiffusion
from torch.optim.lr_scheduler import MultiStepLR
import gc 
from torch.cuda.amp import autocast
import shutil

##Default distributed training parameters
# DEFAULT_MASTER_ADDR = "127.0.0.1"
# DEFAULT_MASTER_PORT = "29500"
# DEFAULT_NPROC_PER_NODE = 3
# DEFAULT_NNODES = 1
# DEFAULT_NODE_RANK = 0

# # Set environment variables for distributed training
# os.environ["MASTER_ADDR"] = DEFAULT_MASTER_ADDR
# os.environ["MASTER_PORT"] = DEFAULT_MASTER_PORT
# os.environ["WORLD_SIZE"] = str(DEFAULT_NPROC_PER_NODE * DEFAULT_NNODES)
# os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
# os.environ["RANK"] = os.environ.get("RANK", "0")

# def delete_folders(base_path, del_indices):
#     # Handle single digits with leading zero
#     folders_to_delete = []
#     for idx in del_indices:
#         if idx < 10:
#             folders_to_delete.append(f'A01_P0{idx}')
#         else:
#             folders_to_delete.append(f'A01_P{idx}')
    
#     for folder in folders_to_delete:
#         folder_path = os.path.join(base_path, folder)
#         if os.path.exists(folder_path):
#             try:
#                 shutil.rmtree(folder_path)
#                 print(f"Successfully deleted {folder}")
#             except Exception as e:
#                 print(f"Error deleting {folder}: {e}")
#         else:
#             print(f"Folder {folder} does not exist")

# base_path = "contrast_normalized_ventricles"
# del_indx = [2,3,4,10,23,26,29,32,53,58,59,65,69,80,138,139,146,147,148,154,155,162,168,169,177,184,185,192,193,200,201,202,203,214,215,218,235,237,238,247,249,250,253,258,259,262,265,268,269,270,274,276,284,292,293,297,298,299,300,301,309,310,317,318,319,326,327,335,336,337,343,344,345,346,356,356,357,366,367,368,376] 
# delete_folders(base_path, del_indx)

torch.cuda.empty_cache()
start = timeit.default_timer()
BATCH_SIZE = 13
MAX_EPOCH = 140
epoch_milestones = [80, 100]
#root_dir = '/data/hfn5052/text2motion/videoflowdiff_natops'
root_dir = 'videoflowdiff_natops'
#data_dir = "/data/hfn5052/text2motion/dataset/NATOPS/split_img_data"
data_dir = 'contrast_normalized_ventricles' #"cropped_ventricles"
mask_dir = "cropped_masks"
mask_dir_dense = 'cropped_masks_dense'
displacement_dir = 'cropped_displacement'   #'displacement_dense'
dense_dir = 'cropped_dense'
GPU = "0,1,2,3"
postfix = "-j-of-lnc-upconv"
joint = "joint" in postfix or "-j" in postfix  # allow joint training with unconditional model
# if "random" in postfix:
#     frame_sampling = "random"
# elif "-vr" in postfix:
#     frame_sampling = "very_random"
# else:
#     frame_sampling = "uniform"

frame_sampling = "very_random"
only_use_flow = "onlyflow" in postfix or "-of" in postfix  # whether only use flow loss
if joint:
    null_cond_prob = 0.1
else:
    null_cond_prob = 0.0
if "upconv" in postfix:
    use_deconv = False
    padding_mode = "reflect"
else:
    use_deconv = True
use_residual_flow = "-rf" in postfix
learn_null_cond = "-lnc" in postfix
#INPUT_SIZE = 128
INPUT_SIZE = 64   #48
#N_FRAMES = 40
N_FRAMES = 10
LEARNING_RATE = 1e-5  #1e-4
RANDOM_SEED = 1234
MEAN = (0.0, 0.0, 0.0)

#RESTORE_FROM = ""
#AE_RESTORE_FROM = "/data/hfn5052/text2motion/RegionMM/log-natops/natops128-crop/snapshots-crop/RegionMM_0100_S024000.pth"
AE_RESTORE_FROM = "LFAE_NATOPS.pth"
#config_pth = "/workspace/code/CVPR23_LFDM/config/natops128.yaml"
config_pth = "config/natops128.yaml"
SNAPSHOT_DIR = os.path.join(root_dir, 'snapshots'+postfix)
IMGSHOT_DIR = os.path.join(root_dir, 'imgshots'+postfix)
VIDSHOT_DIR = os.path.join(root_dir, "vidshots"+postfix)
SAMPLE_DIR = os.path.join(root_dir, 'sample'+postfix)
NUM_EXAMPLES_PER_EPOCH = 4800
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
SAVE_MODEL_EVERY = NUM_STEPS_PER_EPOCH * (MAX_EPOCH // 4)
SAVE_VID_EVERY = 100  #200
SAMPLE_VID_EVERY = 100 #800
UPDATE_MODEL_EVERY = 400

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(IMGSHOT_DIR, exist_ok=True)
os.makedirs(VIDSHOT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

LOG_PATH = SNAPSHOT_DIR + "/B"+format(BATCH_SIZE, "04d")+"E"+format(MAX_EPOCH, "04d")+".log"
sys.stdout = Logger(LOG_PATH, sys.stdout)
print(root_dir)
print("update saved model every:", UPDATE_MODEL_EVERY)
print("save model every:", SAVE_MODEL_EVERY)
print("save video every:", SAVE_VID_EVERY)
print("sample video every:", SAMPLE_VID_EVERY)
print(postfix)
print("num examples per epoch:", NUM_EXAMPLES_PER_EPOCH)
print("max epoch:", MAX_EPOCH)
print("image size, num frames:", INPUT_SIZE, N_FRAMES)
print("epoch milestones:", epoch_milestones)
print("frame sampling:", frame_sampling)
print("only use flow loss:", only_use_flow)
print("null_cond_prob:", null_cond_prob)
print("use residual flow:", use_residual_flow)
print("learn null cond:", learn_null_cond)
print("use deconv:", use_deconv)

# get the latest checkpoint
def get_latest_checkpoint(snapshot_dir):
    """Find the latest checkpoint in the snapshot directory."""
    if not os.path.exists(snapshot_dir):
        return None, 0
    
    # Find all checkpoint files
    checkpoints = []
    for f in os.listdir(snapshot_dir):
        if f.startswith('flowdiff_') and f.endswith('.pth'):
            try:
                # Extract step number from filename
                step = int(f.split('_S')[-1].split('.pth')[0])
                checkpoints.append((f, step))
            except:
                continue
    
    if not checkpoints:
        return None, 0
        
    # Sort by step number and return the latest
    latest_checkpoint = max(checkpoints, key=lambda x: x[1])
    return os.path.join(snapshot_dir, latest_checkpoint[0]), latest_checkpoint[1]

# Find latest checkpoint
latest_checkpoint, start_step = get_latest_checkpoint(SNAPSHOT_DIR)
RESTORE_FROM = latest_checkpoint
#RESTORE_FROM = 'videoflowdiff_natops/snapshots-j-of-lnc-upconv/flowdiff_0013_S349700.pth'
set_start = start_step
print("RESTORE_FROM", RESTORE_FROM)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """

    # args = get_arguments()
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--set-start", default=True)  #default=False
    parser.add_argument("--start-step", default=set_start, type=int)  #0
    parser.add_argument("--img-dir", type=str, default=IMGSHOT_DIR,
                        help="Where to save images of the model.")
    parser.add_argument("--num-workers", default=4)   #8,4
    # parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
    #                     help="Number of training steps.")
    parser.add_argument("--final-step", type=int, default=3360000000,
                        help="Number of training steps.")
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=2, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--save-img-freq', default=20, type=int,
                        metavar='N', help='save image frequency')
    parser.add_argument('--save-vid-freq', default=SAVE_VID_EVERY, type=int)
    parser.add_argument('--sample-vid-freq', default=SAMPLE_VID_EVERY, type=int)
    parser.add_argument('--evaluation-step', default=1, type=int)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--n-frames", default=N_FRAMES)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    parser.add_argument("--save-pred-every", type=int, default=SAVE_MODEL_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--update-pred-every", type=int, default=UPDATE_MODEL_EVERY)
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--fp16", default=False)
    return parser.parse_args()

args = get_arguments()


def sample_img(rec_img_batch, idx=0):
    rec_img = rec_img_batch[idx].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)

def load_pretrained_weights(model, checkpoint, verbose=True):
    """
    Load pretrained weights from old architecture to new architecture.
    Handles both DataParallel and non-DataParallel cases.
    
    Args:
        model: New model architecture
        checkpoint: Loaded checkpoint dictionary
        verbose: Whether to print loading info
    """
    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.diffusion.state_dict()
        old_state_dict = checkpoint['diffusion']
    else:
        model_state_dict = model.module.diffusion.state_dict()
        old_state_dict = checkpoint['diffusion'] 

    # Create dict to store weights that were loaded and initialized
    loaded_weights = []
    initialized_weights = []

    # Load existing weights and initialize new ones
    for name, param in model_state_dict.items():
        if name in old_state_dict:
            try:
                # Load existing weight
                param.copy_(old_state_dict[name])
                loaded_weights.append(name)
            except:
                torch.nn.init.kaiming_normal_(param)
        else:
            # Initialize new weight using default initialization
            if isinstance(param, torch.nn.Parameter):
                if len(param.data.shape) == 1:
                    param_reshaped = param.data.view(1, -1)
                    torch.nn.init.kaiming_normal_(param_reshaped)
                    param.data = param_reshaped.view(-1) 
                else:
                    torch.nn.init.kaiming_normal_(param.data)
            initialized_weights.append(name)

    if verbose:
        print(f"\nLoaded {len(loaded_weights)} weights from old architecture")
        print(f"Initialized {len(initialized_weights)} new weights")
        
        if initialized_weights:
            print("\nNewly initialized layers:")
            for name in initialized_weights:
                print(f"- {name}")

    return loaded_weights, initialized_weights

def setup_data_loader(data_dir, mask_dir, input_size, n_frames, batch_size, num_workers, mean):
    # Create the dataset
    dataset = NATOPS(data_dir=data_dir,
                    mask_dir=mask_dir,
                    image_size=input_size,
                    num_frames=n_frames,
                    color_jitter=True,
                    sampling=frame_sampling,
                    mean=mean)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),  # Total number of GPUs
        rank=dist.get_rank(),                # Current GPU rank
        shuffle=True                         # Whether to shuffle the data
    )
    
    # Create the dataloader with the distributed sampler
    trainloader = data.DataLoader(
        dataset,
        batch_size=batch_size,               # This is the batch size per GPU
        shuffle=False,                       # Don't shuffle - sampler will do it
        num_workers=num_workers,
        sampler=sampler,                     # Use the distributed sampler
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True
    )
    
    return trainloader, sampler

def init_distributed():
    dist.init_process_group(backend='nccl')  # Use NCCL for GPU communication
    #local_rank = torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    return local_rank
    
def save_sampled_images(images, sequence_name, base_dir='sampled_images'):
    """
    Save a sequence of images to a directory structure.
    
    Args:
        images: List of numpy arrays or tensor containing the image sequence
        sequence_name: Name of the sequence (will be used as subdirectory name)
        base_dir: Base directory for saving images
    """
    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create sequence-specific subdirectory
    sequence_dir = os.path.join(base_dir, sequence_name)
    os.makedirs(sequence_dir, exist_ok=True)
    
    # Convert images to numpy arrays if they're tensors
    if torch.is_tensor(images):
        if len(images.shape) == 5:  # [B,C,T,H,W] format
            images = images.squeeze(0).permute(1,0,2,3).cpu().numpy()  # [T,C,H,W]
        elif len(images.shape) == 4:  # [C,T,H,W] format
            images = images.permute(1,0,2,3).cpu().numpy()  # [T,C,H,W]
    
    # Save each image in the sequence
    for i, img in enumerate(images):
        if len(img.shape) == 3:  # If image has channel dimension
            img = img.squeeze()  # Remove channel dimension for grayscale
            
        # Ensure pixel values are in [0, 255] range
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
            #mask = img > 215
            #img[mask] = 0
        else:
            img = img.astype(np.uint8)
            #mask = img > 215
            #img[mask] = 0
            
        # Create image file name
        img_name = f'image_{i:04d}.png'
        img_path = os.path.join(sequence_dir, img_name)
        
        # Save the image
        Image.fromarray(img).save(img_path)


def main():
    """Create the model and start the training."""
    
    local_rank = init_distributed()  # Initialize process group
    is_main_process = local_rank == 0  # Flag for main process
    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)

    model = FlowDiffusion(is_train=True,
                         img_size=INPUT_SIZE,
                         num_frames=N_FRAMES,
                         lr=args.learning_rate,
                         null_cond_prob=null_cond_prob,
                         sampling_timesteps=1000,
                         use_residual_flow=use_residual_flow,
                         learn_null_cond=learn_null_cond,
                         use_deconv=use_deconv,
                         padding_mode=padding_mode,
                         config_pth=config_pth,
                         pretrained_pth=AE_RESTORE_FROM)
    
    # Move model to GPU
    model.cuda()
    
    ###Optionally wrap model in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        #model = torch.nn.DataParallel(model)
        model = DDP(model)
        model.module.diffusion.use_checkpoint = True  # If using DDP (your case)

    if args.fine_tune:
        pass
    elif args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from)
            if args.set_start:
                args.start_step = int(math.ceil(checkpoint['example'] / args.batch_size))

            loaded_weights, initialized_weights = load_pretrained_weights(model, checkpoint)

            # Load optimizer state if it exists and start step is set
            if args.set_start and "optimizer_diff" in checkpoint:
                try:
                    model.module.optimizer_diff.load_state_dict(checkpoint['optimizer_diff'])
                    # Update learning rate
                    for param_group in model.module.optimizer_diff.param_groups:
                        #param_group['initial_lr'] = args.learning_rate
                        param_group['lr'] = args.learning_rate
                    print("=> loaded optimizer state")
                except:
                    print("=> could not load optimizer state, initializing optimizer")
                    
            print("=> loaded checkpoint '{}' (step {})".format(
                args.restore_from, args.start_step))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))
    else:
        print("NO checkpoint found!")

    setup_seed(args.random_seed)
    # trainloader = data.DataLoader(NATOPS(data_dir=data_dir,
    #                                    mask_dir=mask_dir,
    #                                    image_size=INPUT_SIZE,
    #                                    num_frames=N_FRAMES,
    #                                    color_jitter=True,
    #                                    sampling=frame_sampling,
    #                                    mean=MEAN),
    #                             batch_size=args.batch_size,
    #                             shuffle=True, num_workers=args.num_workers,
    #                             prefetch_factor=2, persistent_workers=True,
    #                             pin_memory=True)

    trainloader, train_sampler = setup_data_loader(
                                    data_dir=data_dir,
                                    mask_dir=mask_dir,
                                    input_size=INPUT_SIZE,
                                    n_frames=N_FRAMES,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    mean=MEAN
                                )

    testloader = data.DataLoader(NATOPS_test(data_dir=data_dir,
                                       mask_dir=mask_dir_dense,
                                       displacement_dir=displacement_dir,
                                       dense_dir=dense_dir,
                                       cine_mask_dir=mask_dir,
                                       image_size=INPUT_SIZE,
                                       num_frames=N_FRAMES,
                                       color_jitter=True,
                                       mean=MEAN),
                                       batch_size=1,
                                       shuffle=True, 
                                       num_workers=args.num_workers,
                                       pin_memory=True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    losses = AverageMeter()
    losses_rec = AverageMeter()
    losses_warp = AverageMeter()
    losses_flow = AverageMeter()
    
    cnt = 0
    actual_step = args.start_step
    #start_epoch = int(math.ceil((args.start_step * args.batch_size)/NUM_EXAMPLES_PER_EPOCH))
    start_epoch = actual_step
    epoch_cnt = start_epoch
    
    for param_group in model.module.optimizer_diff.param_groups:
        param_group['initial_lr'] = args.learning_rate

    #scheduler = MultiStepLR(model.optimizer_diff, epoch_milestones, gamma=0.1, last_epoch=start_epoch - 1)
    scheduler = MultiStepLR(model.module.optimizer_diff, epoch_milestones, gamma=1, last_epoch=start_epoch - 1)
    print("epoch %d, lr= %.7f" % (epoch_cnt, model.module.optimizer_diff.param_groups[0]["lr"]))
    
    while actual_step < args.final_step:
        iter_end = timeit.default_timer()
        for i_iter, batch in enumerate(trainloader):
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            actual_step = int(args.start_step + cnt)
            data_time.update(timeit.default_timer() - iter_end)
            
            real_vids, real_mask, ref_texts, real_names = batch
            # use first frame of each video as reference frame
            ref_imgs = real_vids[:, :, 0, :, :].clone().detach()
            bs = real_vids.size(0)
            
            model.module.set_train_input(ref_img=ref_imgs, real_vid=real_vids, real_mask=real_mask, ref_text=ref_texts)
            model.module.optimize_parameters()
            
            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()
            
            losses.update(model.module.loss, bs)
            losses_rec.update(model.module.rec_loss, bs)
            #losses_warp.update(model.rec_warp_loss, bs)
            losses_flow.update(model.module.flow_loss, bs)
            
            if actual_step % args.print_freq == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'loss {loss.val:.7f} ({loss.avg:.7f})\t'
                      'loss_rec {loss_rec.val:.4f} ({loss_rec.avg:.4f})\t'
                      'loss_warp {loss_warp.val:.4f} ({loss_warp.avg:.4f})'
                      #'loss_warp {flow_loss.val:.4f} ({flow_loss.avg:.4f})'
                    .format(
                    cnt, actual_step, args.final_step,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_rec=losses_rec,
                    #loss_warp=losses_warp,
                    loss_warp=losses_flow,
                ))

            null_cond_mask = np.array(model.module.diffusion.denoise_fn.null_cond_mask.data.cpu().numpy(),
                                      dtype=np.uint8)
            
            # First add the directory creation at the top of your script with other makedirs:
            CINE_MOTION_DIR = os.path.join(root_dir, 'Cine_Motion_Fake')
            os.makedirs(CINE_MOTION_DIR, exist_ok=True)
            real_CINE_MOTION_DIR = os.path.join(root_dir, 'Cine_Motion_real')
            os.makedirs(real_CINE_MOTION_DIR, exist_ok=True)
            CINE_mask_DIR = os.path.join(root_dir, 'Cine_mask')
            os.makedirs(CINE_mask_DIR, exist_ok=True)
            dense_mask_DIR = os.path.join(root_dir, 'dense_mask')
            os.makedirs(dense_mask_DIR, exist_ok=True)
            dense_DIR = os.path.join(root_dir, 'dense_motion')
            os.makedirs(dense_DIR, exist_ok=True)

            if actual_step % args.save_vid_freq == 0:
                print("saving video...")
                num_frames = real_vids.size(2)
                msk_size = ref_imgs.shape[-1]
                new_im_arr_list = []
                fake_motion_arr_list = []  # New list for motion-only frames
                real_motion_arr_list = []  # For real motion
                save_src_img = sample_img(ref_imgs)
                for nf in range(num_frames):
                    save_tar_img = sample_img(real_vids[:, :, nf, :, :])
                    save_real_out_img = sample_img(model.module.real_out_vid[:, :, nf, :, :])
                    save_fake_out_img = sample_img(model.module.fake_out_vid[:, :, nf, :, :])
                    new_im = Image.new('L', (msk_size * 5, msk_size * 2))
                    new_im.paste(Image.fromarray(save_src_img.squeeze(), 'L'), (0, 0))
                    new_im.paste(Image.fromarray(save_tar_img.squeeze(), 'L'), (0, msk_size))
                    new_im.paste(Image.fromarray(save_real_out_img.squeeze(), 'L'), (msk_size, 0))
                    new_im.paste(Image.fromarray(save_fake_out_img.squeeze(), 'L'), (msk_size * 2, 0))
                    new_im_arr = np.array(new_im)
                    new_im_arr_list.append(new_im_arr)
                    
                    # Save just the fake_out_vid frame
                    fake_motion_arr_list.append(save_fake_out_img.squeeze())
                    real_motion_arr_list.append(save_real_out_img.squeeze())

                new_vid_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                               + '_' + real_names[0] + "_%d.gif" % (null_cond_mask[0])
                new_vid_file = os.path.join(VIDSHOT_DIR, new_vid_name)
                imageio.mimsave(new_vid_file, new_im_arr_list, loop=0, duration=500)

            # sampling
            if actual_step % args.sample_vid_freq == 0:
                ## save model at i-th step
                print('taking snapshot ...')
                save_dict = {
                    'example': actual_step * args.batch_size,
                    'diffusion': model.module.diffusion.state_dict() if isinstance(model, torch.nn.DataParallel) else model.module.diffusion.state_dict(),
                    'optimizer_diff': model.module.optimizer_diff.state_dict()
                }
                torch.save(save_dict, osp.join(args.snapshot_dir,
                                'flowdiff_' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))

                # update saved model
                print('updating saved snapshot ...')
                save_dict = {
                    'example': actual_step * args.batch_size,
                    'diffusion': model.module.diffusion.state_dict() if isinstance(model, torch.nn.DataParallel) else model.module.diffusion.state_dict(),
                    'optimizer_diff': model.module.optimizer_diff.state_dict()
                }
                torch.save(save_dict, osp.join(args.snapshot_dir, 'flowdiff.pth'))

                print("sampling video...")
                batch = next(iter(testloader))
                real_vids, real_mask, dense_displacement, real_dense, cine_mask, ref_texts, real_names = batch
                dense_displacement = dense_displacement.squeeze(1)
                ref_imgs = real_vids[:, :, 0, :, :].clone().detach()
                with torch.no_grad():
                    model.module.set_train_input(ref_img=ref_imgs, real_vid=real_vids, real_mask=real_mask, ref_text=ref_texts)
                    model.module.forward()
                model.module.set_sample_input(sample_img=ref_imgs, displacement = dense_displacement, mask=real_mask, dense=real_dense, name=real_names[0],
                                       sample_text=[ref_texts[0]])
                model.module.sample_one_video(cond_scale=1.0)

                # ###Save the sampled video frames
                # for i, name in enumerate(real_names):
                #     # Save original video frames
                #     save_sampled_images(
                #         real_vids[i], 
                #         f"{name}_original", 
                #         base_dir='sampled_images'
                #     )
                    
                #     # Save generated video frames
                #     save_sampled_images(
                #         model.module.sample_out_vid[i], 
                #         f"{name}_generated", 
                #         base_dir='sampled_images'
                #     )

                num_frames = real_vids.size(2)
                msk_size = ref_imgs.shape[-1]
                new_im_arr_list = []
                fake_motion_arr_list = []  # New list for motion-only frames
                dense_arr_list = []
                mask_arr_list = []
                real_vid_arr_list = []
                cine_mask_arr_list = []
                save_src_img = sample_img(ref_imgs)
                for nf in range(num_frames):
                    save_tar_img = sample_img(real_vids[:, :, nf, :, :])
                    save_mask_img = sample_img(real_mask[:, :, nf, :, :])
                    save_cine_mask_img = sample_img(cine_mask[:, :, nf, :, :])
                    save_dense_img = sample_img(real_dense[:, :, nf, :, :])
                    save_real_out_img = sample_img(model.module.real_out_vid[:, :, nf, :, :])
                    save_sample_out_img = sample_img(model.module.sample_out_vid[:, :, nf, :, :])
                    new_im = Image.new('L', (msk_size * 5, msk_size * 2))
                    new_im.paste(Image.fromarray(save_src_img.squeeze(), 'L'), (0, 0))
                    new_im.paste(Image.fromarray(save_tar_img.squeeze(), 'L'), (0, msk_size))
                    new_im.paste(Image.fromarray(save_dense_img.squeeze(), 'L'), (msk_size, msk_size))
                    new_im.paste(Image.fromarray(save_mask_img.squeeze(), 'L'), (msk_size*2, msk_size))
                    new_im.paste(Image.fromarray(save_real_out_img.squeeze(), 'L'), (msk_size, 0))

                    new_im.paste(Image.fromarray(save_sample_out_img.squeeze(), 'L'), (msk_size * 2, 0))
                    try:
                        deform_grid = Image.open('test_deformation.gif')
                        if hasattr(deform_grid, 'seek'):
                            deform_grid.seek(nf % deform_grid.n_frames)
                        
                        # Resize to use full height (2 * msk_size) while maintaining aspect ratio
                        full_height = msk_size * 2
                        deform_grid = deform_grid.resize((full_height, full_height))
                        
                        # Paste the large deformation grid in the rightmost column
                        new_im.paste(deform_grid.convert('L'), (msk_size * 3, 0))
                        
                    except Exception as e:
                        print(f"Could not load deformation grid: {e}")

                    new_im_arr = np.array(new_im)
                    new_im_arr_list.append(new_im_arr)
                    fake_motion_arr_list.append(save_sample_out_img.squeeze())
                    dense_arr_list.append(save_dense_img.squeeze())
                    mask_arr_list.append(save_mask_img.squeeze())
                    real_vid_arr_list.append(save_tar_img.squeeze())
                    cine_mask_arr_list.append(save_cine_mask_img.squeeze())

                    new_vid_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                                + '_' + real_names[0] + ".gif"
                    new_vid_file = os.path.join(SAMPLE_DIR, new_vid_name)
                    imageio.mimsave(new_vid_file, new_im_arr_list, loop=0, duration=500)

                    metrics = calculate_all_metrics(model.module.sample_out_vid, real_vids.cuda())
                    print(f"PSNR: {metrics['psnr']:.4f}")
                    print(f"SSIM: {metrics['ssim']:.4f}")
                    print(f"MSE: {metrics['mse']:.4f}\n")
                    print(f"LPIPS: {metrics['lpips']:.4f}")
                    print(f"FloLPIPS: {metrics['flolpips']:.4f}\n")
            

                if actual_step >= args.final_step:
                    break

                #cnt += 1
            cnt += 1
            dist.barrier()
            
            # if actual_step % args.evaluation_step == 0:
            #     print("Evaluating model...")
            #     metrics = evaluate_and_log(model, testloader, actual_step, args)
            
            scheduler.step()
            epoch_cnt += 1
            print("epoch %d, lr= %.7f" % (epoch_cnt, model.module.optimizer_diff.param_groups[0]["lr"]))

        if actual_step % args.sample_vid_freq == 0:
            print('save the final model ...')
            save_dict = {
                'example': actual_step * args.batch_size,
                'diffusion': model.module.diffusion.state_dict() if isinstance(model, torch.nn.DataParallel) else model.module.diffusion.state_dict(),
                'optimizer_diff': model.module.optimizer_diff.state_dict()
            }
            torch.save(save_dict, osp.join(args.snapshot_dir,
                            'flowdiff_' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))
            end = timeit.default_timer()
            print(end - start, 'seconds')
        torch.cuda.empty_cache()
        gc.collect()

def evaluate_and_log(model, testloader, actual_step, args):
    """Helper function to evaluate and log metrics"""
    metrics = evaluate_model(model, testloader)
    print(f"\nStep {actual_step} Evaluation Metrics:")
    print(f"PSNR: {metrics['psnr']:.4f}")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}\n")
    print(f"LPIPS: {metrics['lpips']:.4f}")
    print(f"FloLPIPS: {metrics['flolpips']:.4f}\n")
    
    # Optionally save metrics to a log file
    log_file = os.path.join(args.snapshot_dir, 'evaluation_metrics.txt')
    with open(log_file, 'a') as f:
        f.write(f"Step {actual_step}:\n")
        f.write(f"PSNR: {metrics['psnr']:.4f}\n")
        f.write(f"SSIM: {metrics['ssim']:.4f}\n")
        f.write(f"MSE: {metrics['mse']:.4f}\n\n")
        f.write(f"LPIPS: {metrics['lpips']:.4f}\n\n")
        f.write(f"FloLPIPS: {metrics['flolpips']:.4f}\n\n")
        
    
    return metrics

def calculate_metrics(pred_frames, target_frames):
    """
    Calculate PSNR, SSIM and MSE for a batch of frames
    
    Args:
        pred_frames: Predicted frames tensor (B, C, T, H, W)
        target_frames: Target frames tensor (B, C, T, H, W)
    
    Returns:
        Dictionary containing average PSNR, SSIM and MSE values
    """
    # Convert to numpy and scale to [0,1]
    pred_np = pred_frames.cpu().numpy().squeeze()  # Remove channel dim for grayscale
    target_np = target_frames.cpu().numpy().squeeze()
    
    if pred_np.ndim == 4:  # If batch dimension exists
        B, T, H, W = pred_np.shape
    else:
        T, H, W = pred_np.shape
        B = 1
        pred_np = pred_np[None,...]
        target_np = target_np[None,...]
    
    # Initialize metrics
    total_psnr = 0
    total_ssim = 0
    total_mse = 0
    
    # Calculate metrics for each batch and time step
    for b in range(B):
        for t in range(T):
            # Get single frame
            pred_frame = pred_np[b,t]
            target_frame = target_np[b,t]
            
            # Normalize to [0,1] if not already
            if pred_frame.max() > 1:
                pred_frame = pred_frame / 255.0
            if target_frame.max() > 1:
                target_frame = target_frame / 255.0
            
            # Calculate MSE
            mse = np.mean((pred_frame - target_frame) ** 2)
            total_mse += mse
            
            # Calculate PSNR
            if mse > 0:
                psnr = 10 * np.log10(1.0 / mse)
            else:
                psnr = 100  # Arbitrary large number for perfect prediction
            total_psnr += psnr
            
            # Calculate SSIM
            ssim_score = ssim(target_frame, pred_frame, data_range=1.0)
            total_ssim += ssim_score
    
    # Calculate averages
    avg_psnr = total_psnr / (B * T)
    avg_ssim = total_ssim / (B * T)
    avg_mse = total_mse / (B * T)
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'mse': avg_mse
    }

class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained VGG16
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # Freeze parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Select layers for feature extraction
        self.layers = {
            '3': 'relu1_2',   # After 2nd conv
            '8': 'relu2_2',   # After 4th conv
            '15': 'relu3_3',  # After 7th conv
            '22': 'relu4_3'   # After 10th conv
        }
        
        # Move to GPU
        self.vgg.cuda()
        self.vgg.eval()

    def get_features(self, x):
        features = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if str(i) in self.layers:
                features[self.layers[str(i)]] = normalize_tensor(x)
        return features

    def forward(self, x, y):
        # Ensure input range is [0, 1]
        if x.min() < 0 or x.max() > 1:
            x = (x + 1) / 2
        if y.min() < 0 or y.max() > 1:
            y = (y + 1) / 2
            
        # Convert grayscale to RGB if necessary
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.size(1) == 1:
            y = y.repeat(1, 3, 1, 1)

        # Get features for both images
        x_features = self.get_features(x)
        y_features = self.get_features(y)
        
        # Calculate perceptual distance
        dist = 0
        for layer in self.layers.values():
            x_feat = x_features[layer]
            y_feat = y_features[layer]
            dist += torch.mean((x_feat - y_feat) ** 2)
            
        return dist

class FloLPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = LPIPS()
        self.flow_weight = 0.5  # Weight for flow-based component

    def forward(self, x_sequence, y_sequence):
        """
        Calculate FloLPIPS distance between two video sequences
        Args:
            x_sequence: (B, C, T, H, W) tensor
            y_sequence: (B, C, T, H, W) tensor
        Returns:
            FloLPIPS distance
        """
        B, C, T, H, W = x_sequence.shape
        total_dist = 0
        
        # Calculate regular LPIPS for each frame
        spatial_dist = 0
        for t in range(T):
            x_frame = x_sequence[:, :, t]
            y_frame = y_sequence[:, :, t]
            spatial_dist += self.lpips(x_frame, y_frame)
        spatial_dist /= T
        
        # Calculate temporal LPIPS using frame differences
        temporal_dist = 0
        if T > 1:
            for t in range(T-1):
                x_diff = x_sequence[:, :, t+1] - x_sequence[:, :, t]
                y_diff = y_sequence[:, :, t+1] - y_sequence[:, :, t]
                temporal_dist += self.lpips(x_diff, y_diff)
            temporal_dist /= (T-1)
        
        # Combine spatial and temporal components
        total_dist = spatial_dist + self.flow_weight * temporal_dist
        return total_dist

def normalize_tensor(x):
    """Normalize feature tensors"""
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + 1e-10)

def calculate_all_metrics(pred_frames, target_frames):
    """
    Calculate all metrics including LPIPS and FloLPIPS
    Args:
        pred_frames: Predicted frames tensor (B, C, T, H, W)
        target_frames: Target frames tensor (B, C, T, H, W)
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Initialize metric modules
    lpips_metric = LPIPS().cuda()
    flolpips_metric = FloLPIPS().cuda()
    
    # Calculate PSNR and SSIM using existing code
    basic_metrics = calculate_metrics(pred_frames, target_frames)
    metrics.update(basic_metrics)
    
    # Calculate LPIPS
    with torch.no_grad():
        B, C, T, H, W = pred_frames.shape
        lpips_values = []
        for t in range(T):
            lpips_val = lpips_metric(pred_frames[:, :, t], target_frames[:, :, t])
            lpips_values.append(lpips_val.item())
        metrics['lpips'] = sum(lpips_values) / len(lpips_values)
        
        # Calculate FloLPIPS
        flolpips_val = flolpips_metric(pred_frames, target_frames)
        metrics['flolpips'] = flolpips_val.item()
    
    return metrics

def evaluate_model(model, testloader, device='cuda'):
    """
    Evaluate model on test set and return average metrics
    
    Args:
        model: The model to evaluate
        testloader: DataLoader containing test data
        device: Device to run evaluation on
    
    Returns:
        Dictionary containing average metrics across test set
    """
    model.eval()
    total_metrics = {'psnr': 0, 'ssim': 0, 'mse': 0, 'lpips': 0, 'flolpips': 0}
    num_samples = 0

    # Create the directory to save evaluation videos
    test_video_dir = "test_video"
    os.makedirs(test_video_dir, exist_ok=True)

    miccai_video_dir = "miccai_video"
    os.makedirs(miccai_video_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in testloader:
            real_vids, real_mask, dense_displacement, real_dense, cine_mask, ref_texts, real_names = batch
            dense_displacement = dense_displacement.squeeze(1)
            ref_imgs = real_vids[:, :, 0, :, :].clone().detach()
            
            if os.path.exists(f'sampled_images/{real_names[0]}_original'):
                print(f"Skipping {real_names[0]}, folder already exists.")
                continue

            # Move to device
            ref_imgs = ref_imgs.to(device)
            real_vids = real_vids.to(device)
            real_mask = real_mask.to(device)
            
            # Get model predictions
            model.module.set_train_input(ref_img=ref_imgs, real_vid=real_vids, real_mask=real_mask, ref_text=ref_texts)
            model.module.forward()
            model.module.set_sample_input(sample_img=ref_imgs, displacement = dense_displacement, mask=real_mask, dense=real_dense, name=real_names[0],
                                       sample_text=[ref_texts[0]])
            model.module.sample_one_video(cond_scale=1.0)

            # Save the combined GIF file
            for i, name in enumerate(real_names):
                num_frames = real_mask.shape[2]
                msk_size = real_mask.shape[-1]
                combined_frames = []
                miccai_frames = []

                for nf in range(num_frames):
                    save_tar_img = sample_img(real_vids[:, :, nf, :, :])
                    save_mask_img = sample_img(real_mask[:, :, nf, :, :])
                    save_dense_img = sample_img(real_dense[:, :, nf, :, :])
                    save_sample_out_img = sample_img(model.module.sample_out_vid[:, :, nf, :, :])

                    # Create a horizontal combination of images
                    combined_image = Image.new('L', (msk_size * 4, msk_size))
                    combined_image.paste(Image.fromarray(save_tar_img.squeeze(), 'L'), (0, 0))
                    combined_image.paste(Image.fromarray(save_mask_img.squeeze(), 'L'), (msk_size, 0))
                    combined_image.paste(Image.fromarray(save_dense_img.squeeze(), 'L'), (msk_size * 2, 0))
                    combined_image.paste(Image.fromarray(save_sample_out_img.squeeze(), 'L'), (msk_size * 3, 0))

                    # this is for miccai demo
                    miccai_image = Image.new('L', (msk_size * 2, msk_size))
                    #combined_image.paste(Image.fromarray(save_tar_img.squeeze(), 'L'), (0, 0))
                    #miccai_image.paste(Image.fromarray(save_mask_img.squeeze(), 'L'), (msk_size, 0))
                    miccai_image.paste(Image.fromarray(save_dense_img.squeeze(), 'L'), (0, 0))
                    miccai_image.paste(Image.fromarray(save_sample_out_img.squeeze(), 'L'), (msk_size, 0))


                    combined_frames.append(np.array(combined_image))
                    miccai_frames.append(np.array(miccai_image))

                # Save the GIF
                gif_name = f"{name}_evaluation.gif"
                gif_path = os.path.join(test_video_dir, gif_name)
                imageio.mimsave(gif_path, combined_frames, loop=0, duration=500)
                print(f"Saved evaluation GIF: {gif_path}")

                # Save the GIF
                gif_name_miccai = f"{name}_evaluation.gif"
                gif_path_miccai = os.path.join(miccai_video_dir, gif_name_miccai)
                imageio.mimsave(gif_path_miccai, miccai_frames, loop=0, duration=300)
                print(f"Saved evaluation GIF: {gif_path_miccai}")

            # Save the sampled video frames
            for i, name in enumerate(real_names):
                print(f'saving the sampled images for {real_names}')
                # Save original video frames
                save_sampled_images(
                    real_vids[i], 
                    f"{name}_original", 
                    base_dir='sampled_images'
                )
                
                # Save generated video frames
                save_sampled_images(
                    model.module.sample_out_vid[i], 
                    f"{name}_generated", 
                    base_dir='sampled_images'
                )
                
                # Save generated video frames
                save_sampled_images(
                    model.module.sample_out_vid[i], 
                    f"{name}", 
                    base_dir='synthetic_images'
                )
            
            # Calculate all metrics including LPIPS and FloLPIPS
            batch_metrics = calculate_all_metrics(model.module.sample_out_vid, real_vids.cuda())

            with open('performance_metrics.txt', 'a') as f:
                f.write(f'{real_names}\n')
                for metric, value in batch_metrics.items():
                    f.write(f'{metric}: {value}\n')
            
            # Accumulate metrics
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]
            num_samples += 1
    
    # Calculate averages
    avg_metrics = {k: v/num_samples for k, v in total_metrics.items()}
    return avg_metrics

class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
