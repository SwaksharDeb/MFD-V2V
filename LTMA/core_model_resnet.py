import torch
import numpy as np
import torch.nn as nn
from TLRN.utils import Svf
import lagomorph as lm


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernal=3, stride=1, padding=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernal, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock_github(nn.Module):  ## resnetblock
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_github, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.2)

        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = None

        self.stride = stride
    def forward(self, x):   #[4, 32, 8, 8]
        identity = x

        out = self.conv1(identity)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
        # return identity   

import torch
import torch.nn as nn

class SNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SNConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        return self.norm(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        
    def forward(self, x):
        B, N, C = x.shape  # Batch, Sequence length, Channel
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class TemporalAttentionLayer(nn.Module):
    def __init__(self, input_channels, num_heads=8):
        super().__init__()
        self.input_channels = input_channels
        self.spatial_dims = None
        
        # Attention layer
        self.attention = MultiHeadSelfAttention(input_channels, num_heads)
        
        # Layer norm
        self.norm = nn.LayerNorm(input_channels)
        
        # Optional: MLP after attention
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, input_channels * 4),
            nn.GELU(),
            nn.Linear(input_channels * 4, input_channels)
        )
        self.norm2 = nn.LayerNorm(input_channels)
        
    def forward(self, features_seq):
        B = features_seq[0].shape[0]  # Batch size
        
        # Store spatial dimensions
        if features_seq[0].dim() == 4:  # For 2D data
            _, C, H, W = features_seq[0].shape
            self.spatial_dims = (H, W)
        
        # Reshape and concatenate sequence
        features_reshaped = []
        for feat in features_seq:
            # Reshape from [B, C, H, W] to [B, H*W, C]
            f = feat.flatten(2).transpose(1, 2)
            features_reshaped.append(f)
        
        x = torch.stack(features_reshaped, dim=1)  # [B, T, H*W, C]
        B, T, L, C = x.shape
        x = x.reshape(B, T*L, C)  # Combine temporal and spatial dimensions
        
        # Apply attention
        x = x + self.attention(self.norm(x))
        x = x + self.mlp(self.norm2(x))
        
        # Reshape back
        x = x.reshape(B, T, L, C)
        
        # Split back into sequence
        output_seq = []
        for t in range(T):
            feat = x[:, t].transpose(1, 2)  # [B, C, H*W]
            if self.spatial_dims:
                feat = feat.reshape(B, C, *self.spatial_dims)
            output_seq.append(feat)
            
        return output_seq


class Net2DResNet(nn.Module):
    def __init__(self, args=None):
        
        # super().__init__()
        super(Net2DResNet, self).__init__()
        self.args = args
        inshape=(self.args.img_size, self.args.img_size)
        
        self.residue_block_num = (self.args.series_len-2) if not self.args.one_residue_block else 1

        nb_features=args.nb_features
        infeats=2
        nb_levels=None
        max_pool=2
        feat_mult=1
        nb_conv_per_level=1
        half_res=False
        
        self.writer = args.writer
        self.TSteps = args.pred_num_steps
        self.img_size = args.img_size
        self.test_img_size = args.test_img_size

        
        self.id = torch.from_numpy(lm.identity((1,2,args.img_size,args.img_size))).cuda()

        # Initialize the temporal attention layer
        # Using the same channel dimension as in the original residual blocks
        self.temporal_attention = TemporalAttentionLayer(
            input_channels=32,  # This should match your network's channel dimension
            num_heads=8  # Can be adjusted based on your needs
        )


        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        self.final_nf = prev_nf


        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.final_nf, ndims, kernel_size=3, padding=1)
        
        for idex in range(len(self.encoder)):
            self.encoder[idex][0].main.name = f'unet-encoder-{idex}-0'
            self.decoder[idex][0].main.name = f'unet-decoder-{idex}-0'
            
        for idex in range(len(self.remaining)):
            self.remaining[idex].main.name = f'unet-remaining-{idex}'
        

        self.flow.name = 'unet-flow'
        self.residue_block = nn.ModuleList()
        for i in range(self.residue_block_num):
            OneResBlock = nn.ModuleList()
            for level in range(self.args.reslevel):
                if (level==0):
                    # OneResBlock.append(BasicBlock_github(inplanes=32, planes=32))
                    OneResBlock.append(BasicBlock_github(inplanes=64, planes=32))
                else:
                    OneResBlock.append(BasicBlock_github(inplanes=32, planes=32))
            self.residue_block.append(OneResBlock)
            self.residue_block[i].name = f'residual-block-{i}'

        self.MSvf = Svf(inshape=(self.args.img_size, self.args.img_size), steps=self.TSteps)


    
    def exec_encoder(self, x):
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        # encoder forward pass
        x_history = [x]

        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)  #100x100 -> 50x50 -> 25x25  ->12x12  ->6x6
                                        #128x128   64x64   32x32   16x16
        return x, x_history

    def exec_decoder(self, x, x_history):
        skip_index = -1
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history[skip_index]], dim=1)
                skip_index -= 1
        for conv in self.remaining:
            x = conv(x)

        x = self.flow(x)  #[20, 16, 64, 64] -> [20, 2, 64, 64]

        if(len(x.shape)==4):
            return x.permute(0,2,3,1)   #-> [20, 64, 64, 2]
        return x.permute(0,2,3,4,1)   #-> [20, 64, 64, 64,3]

    def pair_register(self, x, masks=None):
        self.src = x[:,0:1,...] #[32, 1, 64, 64]
        self.tar = x[:,1:2,...] #[32, 1, 64, 64]
        if masks is not None:
            self.src_mask = masks[:,0:1,...]

        low_dim_features,x_his = self.exec_encoder(x)  #[b, 32, 8, 8]
        fnow_full_dim = self.exec_decoder(low_dim_features,x_his) #[b, 64, 64, 2]
        v = torch.permute(fnow_full_dim, [0, 3, 1, 2]) #[b, 64, 64, 2] -> [b, 2, 64, 64]

        u, u_seq = self.MSvf(v) #u:[42, 2, 128, 128]
        
        Sdef,phiinv = self.MSvf.transformer(self.src, u)   #sdef: [32, 1, 64, 64]  phiinv:[42, 128, 128, 2]
       
        if masks is not None:
            Sdef_mask, _ = self.MSvf.transformer(self.src_mask, u, mode="nearest")
        else:
            Sdef_mask = None
        
        u_inv, _ = self.MSvf(-v)
        _,phi = self.MSvf.transformer(self.src, u_inv)

        return Sdef, v, [phiinv], Sdef_mask, [phi]
   
    def sequence_register_no_avg_lowf_addlatentf(self, input, masks=None):  #input   series [4, 9, 64, 64]  [32, 3, 128, 128]
        b, seq_num, h, w = input.shape
        self.src = input[:,0:1,...]     #[4, 1, 64, 64]
        if masks is not None:
            self.src_mask = masks[:,0:1,...]


        low_dim_features_seq = []; x_history_seq = []; low_dim_features_merge_seq = []
        v_series = []; Sdef_series = []
        u_series = []; Sdef_mask_series = []
        ui_series = []

        ### Encoder Block
        for cnt  in range(1, seq_num):  #1
            self.second = input[:,cnt:cnt+1,...]     #[4, 1, 64, 64]
            firt_input = torch.cat((self.src, self.second), dim=1)
            low_dim_features, x_his = self.exec_encoder(firt_input)  #[b, 32, 8, 8]

            low_dim_features_seq.append(low_dim_features)   #length: 8
            x_history_seq.append(x_his)
       
        ## Decoder Block
        for cnt in range(0, seq_num-1):  #seq_num-1
            full_dim_features = self.exec_decoder(low_dim_features_merge_seq[cnt], x_history_seq[cnt])   #[4, 64, 64, 2]
            full_dim_features = full_dim_features.permute(0,3,1,2) #[4, 2, 64, 64]

            if "SM" in self.args.dataset:
                full_dim_features = self.blurred_sum(full_dim_features)

            v_series.append(full_dim_features)

            u,_ = self.MSvf(full_dim_features)
            Sdef, phiinv = self.MSvf.transformer(self.src, u)    #[4, 1, 64, 64]

            ui,_ = self.MSvf(-full_dim_features)
            _, phi = self.MSvf.transformer(self.src, ui)    #[4, 1, 64, 64]

            if masks is not None:
                Sdef_mask, _ = self.MSvf.transformer(self.src_mask, u, mode="nearest")
                Sdef_mask_series.append(Sdef_mask)

            Sdef_series.append(Sdef)
            u_series.append(phiinv)
            ui_series.append(phi)

        if masks is None:
            Sdef_mask_series = None
        
        return Sdef_series, v_series, u_series, Sdef_mask_series, ui_series

    def forward(self, x, resmode = "pair", masks=None): #pair  x:[25, 2, 32, 32]   series [4, 9, 64, 64]
        if resmode == 'voxelmorph':
            return self.pair_register(x, masks=masks)
        elif resmode == "TLRN":
            return self.sequence_register_no_avg_lowf_addlatentf(x, masks=masks)
