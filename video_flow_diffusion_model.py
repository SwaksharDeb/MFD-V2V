# use diffusion model to generate pseudo ground truth flow volume based on RegionMM
# 3D noise to 3D flow
# flow size: 2*32*32*40
# some codes based on https://github.com/lucidrains/video-diffusion-pytorch

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from LFAE.modules.generator import Generator
from LFAE.modules.bg_motion_predictor import BGMotionPredictor
from LFAE.modules.region_predictor import RegionPredictor
from DM.modules.video_flow_diffusion import Unet3D, GaussianDiffusion
import yaml
import animation
from tensorboardX import SummaryWriter
import argparse
from torch.cuda.amp import autocast, GradScaler
from TLRN.core_model_resnet import Net2DResNet   #skip-connect
import torchvision
import numpy as np

class FlowDiffusion(nn.Module):
    def __init__(self, img_size=32, num_frames=40, sampling_timesteps=250,
                 null_cond_prob=0.1, ddim_sampling_eta=1., timesteps=1000,
                 dim_mults=(1, 2, 4, 8),
                 lr=1e-4, adam_betas=(0.9, 0.99), is_train=True,
                 only_use_flow=True,
                 use_residual_flow=False,
                 learn_null_cond=False,
                 use_deconv=True,
                 padding_mode="zeros",
                 pretrained_pth="",
                 config_pth=""):
        super(FlowDiffusion, self).__init__()
        self.use_residual_flow = use_residual_flow
        self.only_use_flow = only_use_flow
        self.counter = 0

        if pretrained_pth != "":
            checkpoint = torch.load(pretrained_pth)
        with open(config_pth) as f:
            config = yaml.safe_load(f)

        model_path = 'models/model.pth'
        parser = argparse.ArgumentParser()
        """ ~~~~~~~~~    basic setting ~~~~~~~~~~~~"""
        parser.add_argument('--mode', type=str, default='test', help='train or test')  #test
        parser.add_argument('--server', type=str, default='My', help='server name')
        parser.add_argument('--debug', type=bool, default=False, help='--')
        parser.add_argument('--seed', type=int, default=1234, help='random seed')
        parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

        """ ~~~~~~~~~    about training phase ~~~~~~~~~~~~"""
        parser.add_argument('--loss_type', type=str, default='MSE', help='experiment_name')
        parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train') 
        parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')   #32
        parser.add_argument('--regis_lr', type=float,  default=0.0005, help='lr of unet')
        parser.add_argument('--regis_stp', type=float,  default=100, help='step of unet')
        parser.add_argument('--regis_gamma', type=float,  default=0.5, help='gamma of unet')
        parser.add_argument('--weight_decay', type=float, default=0, help='--')
        parser.add_argument('--svf_reg_w2', type=float, default='0.03', help='--')           ### noskip + noavg
        parser.add_argument('--svf_simi_w', type=float, default=1.0, help='--')
        parser.add_argument('--svf_reg_w', type=float, default=0.03, help='--')
        parser.add_argument('--pred_num_steps', type=int, default=99999, help='5')

        """ ~~~~~~~~~    network setting ~~~~~~~~~~~~"""
        parser.add_argument('--series_len', type=int, default=10, help='3')
        parser.add_argument('--one_residue_block', type=bool, default=False, help='3')  
        parser.add_argument('--reslevel', type=int, default=3, help='--')

        """ ~~~~~~~~~    about testing phase ~~~~~~~~~~~~"""
        parser.add_argument('--visdir', type=str, default='1', help='visdir') 
        parser.add_argument('--testlen', type=int, default=20, help='experiment_name')
        parser.add_argument('--num_steps', type=int, default=7, help='5')

        """ ~~~~~~~~~    import arguments~~~~~~~~~~~~"""
        parser.add_argument('--module_name', type=str, default='resnet', help='the name of module')
        parser.add_argument('--resmode', type=str, default='TLRN', help='TLRN or voxemorph')
        parser.add_argument('--dataset', type=str, default='cine_slice_img_reversed', help='experiment_name')   #dense_addinfull
        parser.add_argument('--test_type', type=str, default='hollowquickdraw', help='pretrain')
        parser.add_argument('--img_size', type=int, default=64, help='input img_size')
        parser.add_argument('--test_img_size', type=int, default=64, help='input img_size')
        parser.add_argument('--databasesize', type=int, default=64, help='input img_size')
        args = parser.parse_args()

        HOME = '/scratch/swd9tc/video_to_video_generation/vdiff_phi_TLRN_MSA/DM' #'/scratch/swd9tc/TLRN-main/TLRN-main'
        dataset_name = args.dataset
        dataset_config = {
        "lemniscate": HOME+ '/datasets/lemniscate_example_series.mat' ,
        "cine_slice_img": HOME+ '/datasets/all_masks.mat' ,
        "reversed_cine_slice_mask": HOME+ '/datasets/all_masks.mat' ,
    }
        if "cine_slice_img_reversed" in dataset_name:
            args.train_dense = dataset_config['cine_slice_img']
            args.test_dense = dataset_config['cine_slice_img']
            args.pathmask = dataset_config['reversed_cine_slice_mask']

        args.nb_features=[[16, 32, 32], [32, 32, 32, 32, 16, 16]]
        args.exp = dataset_name + str(args.img_size)
        #set the dirctory to save the model
        args.HOME = HOME
        snapshot_path =  HOME+"/TLRN/models/{}/".format(args.exp)    
        snapshot_path += args.module_name
        snapshot_path = snapshot_path + '_' + str(args.loss_type) if args.loss_type == 'NCC' else snapshot_path
        snapshot_path = snapshot_path + '_Tsteps' + str(args.num_steps)
        snapshot_path = snapshot_path + '_epo' +str(args.max_epochs)
        snapshot_path = snapshot_path + '_regis_lr' + str(args.regis_lr)
        snapshot_path = snapshot_path + "_weight_decay_" + str(args.weight_decay)
        snapshot_path = snapshot_path + "_resmode" + str(args.resmode)
        snapshot_path = snapshot_path + "_one_residue_block" if args.one_residue_block else snapshot_path
        snapshot_path = snapshot_path + "_reslevel_" + str(args.reslevel)
        snapshot_path = snapshot_path + "_series_len_" + str(args.series_len) if args.series_len !=0 else snapshot_path
        snapshot_path = snapshot_path + "_svf_simi_w_" + str(args.svf_simi_w) if args.svf_simi_w !=1.0 else snapshot_path
        snapshot_path = snapshot_path + "_svf_reg_w_" + str(args.svf_reg_w) if args.svf_reg_w !=1.0 else snapshot_path
        snapshot_path = snapshot_path + "_svf_reg_w2_" + str(args.svf_reg_w2) if args.svf_reg_w2 !=1.0 else snapshot_path

        args.snapshot_path = snapshot_path
        writerdir = snapshot_path.split('/'); writerdir.insert(-1,'log'); writerdir='/'.join(writerdir)
        writer = SummaryWriter(writerdir)
        args.writer = writer
        module_name = args.module_name
        netdic = {'resnet': Net2DResNet(args).cuda()}
        self.net = netdic[module_name]
        self.net.load_state_dict(torch.load(model_path,map_location='cpu')['registration'])
        self.net.eval()

        #inshape = [48,48]
        inshape = [64,64]
        #self.integrate = vxm.layers.VecInt(inshape, 7)
        #self.transformer = vxm.layers.SpatialTransformer(inshape)

        self.generator = Generator(num_regions=config['model_params']['num_regions'],
                                   num_channels=config['model_params']['num_channels'],
                                   revert_axis_swap=config['model_params']['revert_axis_swap'],
                                   **config['model_params']['generator_params']).cuda()
        if pretrained_pth != "":
            self.generator.load_state_dict(checkpoint['generator'])
            self.generator.eval()
            self.set_requires_grad(self.generator, False)

        self.region_predictor = RegionPredictor(num_regions=config['model_params']['num_regions'],
                                                num_channels=config['model_params']['num_channels'],
                                                estimate_affine=config['model_params']['estimate_affine'],
                                                **config['model_params']['region_predictor_params']).cuda()
        if pretrained_pth != "":
            self.region_predictor.load_state_dict(checkpoint['region_predictor'])
            self.region_predictor.eval()
            self.set_requires_grad(self.region_predictor, False)

        self.bg_predictor = BGMotionPredictor(num_channels=config['model_params']['num_channels'],
                                              **config['model_params']['bg_predictor_params'])
        if pretrained_pth != "":
            self.bg_predictor.load_state_dict(checkpoint['bg_predictor'])
            self.bg_predictor.eval()
            self.set_requires_grad(self.bg_predictor, False)

        self.unet = Unet3D(dim=64,
                           channels=1,
                           out_grid_dim=1,
                           out_conf_dim=1,
                           dim_mults=dim_mults,
                           use_bert_text_cond=True,
                           learn_null_cond=learn_null_cond,
                           use_final_activation=False,
                           use_deconv=use_deconv,
                           padding_mode=padding_mode)

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=img_size,
            num_frames=num_frames,
            sampling_timesteps=sampling_timesteps,
            timesteps=timesteps,  # number of steps
            loss_type='l2',  # L1 or L2
            use_dynamic_thres=True,
            null_cond_prob=null_cond_prob,
            ddim_sampling_eta=ddim_sampling_eta,
        )

        self.ref_img = None
        self.ref_img_fea = None
        self.real_vid = None
        self.real_out_vid = None
        self.real_warped_vid = None
        self.real_vid_grid = None
        self.real_vid_conf = None
        self.velocity_grid = None
        self.fake_vel_grid = None

        self.fake_out_vid = None
        self.fake_warped_vid = None
        self.fake_vid_grid = None
        self.fake_vid_conf = None

        self.sample_out_vid = None
        self.sample_warped_vid = None
        self.sample_vid_grid = None
        self.sample_vid_conf = None
        self.scaler = torch.cuda.amp.GradScaler()

        # training
        self.is_train = is_train
        if self.is_train:
            self.unet.train()
            self.diffusion.train()
            self.lr = lr
            self.loss = torch.tensor(0.0).cuda()
            self.rec_loss = torch.tensor(0.0).cuda()
            self.rec_warp_loss = torch.tensor(0.0).cuda()
            self.optimizer_diff = torch.optim.Adam(self.diffusion.parameters(),
                                                   lr=lr, betas=adam_betas)

    def deformation(self, cine_img, phi):
        # Denormalize
        if len(cine_img.shape) > 3:
            cine_img = cine_img[:,0,...]
        grad_source = torch.gradient(cine_img, dim=(-2,-1))  # Returns tuple of (dy, dx)
        grad_source = torch.stack(grad_source, dim=1)  # batch_size x 2 x 48 x 48
        deformed = self.transformer(cine_img.unsqueeze(1), phi)
        return deformed


    def forward(self):
        # compute pseudo ground-truth flow
        with torch.cuda.amp.autocast():
            b, _, nf, H, W = self.real_vid.size()

            real_grid_list = []
            real_conf_list = []
            real_out_img_list = []
            real_warped_img_list = []
            momentum_fields = []
            momentum_fields_mask = []
            flow_fields = []
            deformed_frames = []
            velocity_list = []
            Sdef_series, v_series, u_series, Sdef_mask_series, ui_series = self.net(self.real_vid.squeeze(1), resmode = "TLRN", masks=self.real_mask.squeeze(1))
            b_ = ui_series[1].shape[0]
            zero_tensor = torch.zeros(b, 64, 64, 2).cuda()
            ui_series.insert(0, zero_tensor)
            self.real_out_vid = self.real_vid
            self.scm_mask = torch.stack(ui_series, dim=1).permute(0,4,1,2,3)
            if self.counter%100 == 0:
                animation.create_grid_visualization(self.scm_mask, 'train_deformation', img_size=64)
            
            if self.is_train:
                if self.use_residual_flow:
                    h, w, = H//4, W//4
                    identity_grid = self.get_grid(b, nf, h, w, normalize=True).cuda()
                    self.loss = self.diffusion(torch.cat((self.real_vid_grid - identity_grid,
                                                        self.real_vid_conf*2-1), dim=1),
                                            self.ref_img_fea,
                                            self.ref_text)
                else:
                    self.loss, self.recon_vid = self.diffusion(self.real_vid, self.ref_img, self.scm_mask, self.real_mask, self.ref_text)
                with torch.no_grad():
                    fake_out_img_list = []
                    fake_warped_img_list = []
                    fake_vel_list = []
                    pred = self.diffusion.pred_x0
                    if self.use_residual_flow:
                        self.fake_vid_grid = pred[:, :2, :, :, :] + identity_grid
                    else:
                        #self.fake_vid_grid = pred[:, :2, :, :, :]
                        self.fake_vid_grid = pred
                    self.fake_out_vid = self.fake_vid_grid
                    self.rec_loss = nn.L1Loss()(self.real_vid, self.fake_out_vid)
                    self.flow_loss = 0
                    self.counter += 1

    def optimize_parameters(self):
        with torch.cuda.amp.autocast():
            self.forward()
        self.optimizer_diff.zero_grad()
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer_diff)
        self.scaler.update()

    def sample_one_video(self, cond_scale):
        self.sample_vid_grid = pred
        nf = self.sample_vid_grid.size(2)
        with torch.no_grad():
            sample_out_img_list = []
            sample_warped_img_list = []
            for idx in range(nf):
                sample_grid = self.sample_vid_grid[:, :, idx, :, :]
                sample_out_img_list.append(sample_grid)
        self.sample_out_vid = torch.stack(sample_out_img_list, dim=2)
    
    def set_train_input(self, ref_img, real_vid, real_mask, ref_text):
        self.ref_img = ref_img.cuda()
        self.real_vid = real_vid.cuda()
        self.ref_text = ref_text
        self.real_mask = real_mask.cuda()
        
    def set_sample_input(self, sample_img, displacement, mask, dense, name, sample_text):
        self.sample_img = sample_img.cuda()
        self.real_mask = mask.cuda()
        self.dense = dense.cuda()
        self.sample_text = sample_text        
        self.scm_mask = (displacement/4.3).cuda()
        animation.create_grid_visualization(self.scm_mask, 'test_deformation', img_size=64)

    def save_mask(self, scm_mask, name_str):
        """
        Save the SCM mask to a specific directory structure based on the name string.
        
        Args:
            scm_mask (torch.Tensor): The mask tensor to save
            name_str (str): Name string in format 'A01_P51'
        """
        # Extract the number after 'P' from the name string
        subfolder_num = name_str.split('_P')[1]
        
        # Create the directory structure
        base_dir = 'register'
        subfolder_path = os.path.join(base_dir, subfolder_num)
        
        # Create directories if they don't exist
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Convert tensor to numpy if it's a torch tensor
        if isinstance(scm_mask, torch.Tensor):
            mask_numpy = scm_mask.cpu().numpy()
        else:
            mask_numpy = scm_mask
            
        # Save the mask as .npy file
        save_path = os.path.join(subfolder_path, f"{subfolder_num}_cine.npy")
        np.save(save_path, mask_numpy)
        
        print(f"Saved mask to {save_path}")

    def print_learning_rate(self):
        lr = self.optimizer_diff.param_groups[0]['lr']
        assert lr > 0
        print('lr= %.7f' % lr)

    def get_grid(self, b, nf, H, W, normalize=True):
        if normalize:
            h_range = torch.linspace(-1, 1, H)
            w_range = torch.linspace(-1, 1, W)
        else:
            h_range = torch.arange(0, H)
            w_range = torch.arange(0, W)
        grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b, 1, 1, 1).flip(3).float()  # flip h,w to x,y
        return grid.permute(0, 3, 1, 2).unsqueeze(dim=2).repeat(1, 1, nf, 1, 1)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    bs = 5
    img_size = 64
    num_frames = 40
    ref_text = ["play basketball"] * bs
    ref_img = torch.rand((bs, 3, img_size, img_size), dtype=torch.float32)
    real_vid = torch.rand((bs, 3, num_frames, img_size, img_size), dtype=torch.float32)
    model = FlowDiffusion(use_residual_flow=False,
                          sampling_timesteps=10,
                          img_size=16,
                          config_pth="/workspace/code/CVPR23_LFDM/config/mug128.yaml",
                          pretrained_pth="")
    model.cuda()
    model.eval()
    model.set_sample_input(sample_img=ref_img, sample_text=ref_text)
    model.sample_one_video(cond_scale=1.0)


