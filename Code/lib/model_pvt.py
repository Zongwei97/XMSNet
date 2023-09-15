import timm
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from Code.lib.Transformer import MaskAttentionTransformerBlock
from timm.models import create_model

import Code.lib.pvt_v2



class DWConv(nn.Module):
    def __init__(self, channel, k=3, d=1, relu=False):
        super(DWConv, self).__init__()
        conv = [
            nn.Conv2d(channel, channel, k, 1, (k//2)*d, d, channel, False),
            nn.BatchNorm2d(channel)
        ]
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)         

                                                        
class LGB(nn.Module):
    def __init__(self,channel):
        super(LGB,self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.conv1_1 = nn.Conv2d(channel, channel*4,kernel_size=1)
        self.convdw_1_1 = nn.Conv2d(channel*4, channel,kernel_size=1)
        self.DWconv = DWConv(channel*4)
    def forward(self, x):
        x = self.conv1_1(x)
        x1,x2,x3,x4 = x.chunk(4, dim=1)
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)                      
        x3 = self.pool3(x3)
        x4 = self.pool4(x4)
        x_dwconv = self.DWconv(torch.cat([x1,x2,x3,x4],dim=1))
        out = self.convdw_1_1(x_dwconv)
        return out
        
        
        
        

class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1)) # 1#nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1)) # 1#nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta =  nn.Parameter(torch.zeros(1, num_channels, 1, 1)) # 1#nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
            
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2,3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            sys.exit()

        gate = torch.tanh(embedding * norm + self.beta)  #1. + torch.tanh(embedding * norm + self.beta) 这里不用 + 1，加了就有问题了最终就变成了 x+ CA*x，我们只要CA

        return  gate

def channel_shuffle(x, groups=6):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool
# def avgpool():
#     pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
#     return pool




class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x





def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)




class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        if len(x.shape) ==4:
            x = x.flatten(2).transpose(1, 2)# [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C]
        x = self.proj(x)
        return x

class MLP_4d(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)# [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C]
        x = self.proj(x)
        x= x.transpose(1,2).view(b, c, h, w)
        return x

class AF(nn.Module):    
    def __init__(self,in_dim, out_dim,X):
        super(AF, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.proj_r = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.proj_d = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)


        in_dim = out_dim
        
        

        self.linear_r = nn.Sequential(MLP(input_dim=in_dim, embed_dim=in_dim//2), MLP(input_dim=in_dim//2, embed_dim=in_dim//4))
        self.linear_d = nn.Sequential(MLP(input_dim=in_dim, embed_dim=in_dim//2), MLP(input_dim=in_dim//2, embed_dim=in_dim//4))
        self.linear_m = nn.Sequential(MLP(input_dim=in_dim//4, embed_dim=in_dim//4))


        self.linear = MLP(input_dim=in_dim*2, embed_dim=in_dim)
        self.linear_f = MLP(input_dim=in_dim*2, embed_dim=in_dim)

        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)   
        
        self.layer_30 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_40 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1) 
        
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim//2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim//2),act_fn,nn.Conv2d(out_dim//2, 1, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(1),act_fn,)        
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim//2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim//2),act_fn,nn.Conv2d(out_dim//2, 1, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(1),act_fn,)        

        self.layer_att = nn.Sequential(nn.Conv2d(6, 2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(2),act_fn,)
        self.layer_att_c = nn.Sequential(nn.Conv2d(out_dim*6, out_dim*2, kernel_size=1),nn.BatchNorm2d(out_dim*2),act_fn,)


        self.layer_ful2 = nn.Sequential(nn.Conv2d(out_dim + X, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.gct = GCT(out_dim)

    def forward(self, rgb, depth,xx):
        rgb = self.proj_r(rgb)
        depth = self.proj_d(depth)
        ################################
        n, c, h, w = rgb.shape

        m_r = self.avg(rgb)
        r_r = rgb - m_r
        v_r = self.linear_r(m_r)

        m_d = self.avg(depth)
        r_d = depth - m_d
        v_d = self.linear_d(m_d)
        
        

        v_mix = self.linear_m(v_r * v_d)
        alpha = self.cos(v_r[:,0,:], v_mix[:,0,:])
        beta = self.cos(v_d[:,0,:], v_mix[:,0,:])
        a_r = (alpha /(alpha + beta)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        b_d = (beta /(alpha + beta)).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        x_rgb = self.layer_10(r_r)
        x_dep = self.layer_20(r_d)
        
        
        avg_r = torch.mean(x_rgb,1).unsqueeze(1)
        avg_d = torch.mean(x_dep,1).unsqueeze(1)

        max_r = torch.max(x_rgb,1)[0].unsqueeze(1)
        max_d = torch.max(x_dep,1)[0].unsqueeze(1)

        att_r = self.layer_11(x_rgb)
        att_d = self.layer_21(x_dep)

        att = self.layer_att(torch.cat((att_r, att_d, max_r, max_d, avg_r, avg_r),dim=1))
        
        x_rgb_c = self.layer_30(r_r)
        x_dep_c = self.layer_40(r_d)
        
        avg_r_c = self.avg(x_rgb_c)
        avg_d_c = self.avg(x_dep_c)
        
        max_r_c = self.max(x_rgb_c)
        max_d_c = self.max(x_dep_c)
        
        gct_r_c = self.gct(x_rgb_c)
        gct_d_c = self.gct(x_dep_c)
             
        
        att_c = channel_shuffle(torch.cat([gct_r_c, gct_d_c ,avg_r_c,avg_d_c, max_r_c, max_d_c],dim=1),6)
        att_c = self.layer_att_c(att_c)
        att_cr, att_cd = att_c.chunk(2, dim=1)
        

        att_r = att[:,0,:,:].unsqueeze(1).sigmoid()#?
        att_dep = att[:,1,:,:].unsqueeze(1).sigmoid()
        out = self.linear_f(torch.cat((rgb * a_r * att_r * att_cr , depth * b_d *att_dep* att_cd ), dim=1)).permute(0,2,1).view(rgb.shape)

        

        out2 = self.layer_ful2(torch.cat([out,xx],dim=1))
         
        return out2

class AF0(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(AF0, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.proj_r = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.proj_d = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)


        in_dim = out_dim
        
        

        self.linear_r = nn.Sequential(MLP(input_dim=in_dim, embed_dim=in_dim//2), MLP(input_dim=in_dim//2, embed_dim=in_dim//4))
        self.linear_d = nn.Sequential(MLP(input_dim=in_dim, embed_dim=in_dim//2), MLP(input_dim=in_dim//2, embed_dim=in_dim//4))
        self.linear_m = nn.Sequential(MLP(input_dim=in_dim//4, embed_dim=in_dim//4))


        self.linear = MLP(input_dim=in_dim*2, embed_dim=in_dim)
        self.linear_f = MLP(input_dim=in_dim*2, embed_dim=in_dim)

        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)   
        
        self.layer_30 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_40 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1) 
        
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim//2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim//2),act_fn,nn.Conv2d(out_dim//2, 1, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(1),act_fn,)        
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim//2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim//2),act_fn,nn.Conv2d(out_dim//2, 1, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(1),act_fn,)        

        self.layer_att = nn.Sequential(nn.Conv2d(6, 2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(2),act_fn,)
        self.layer_att_c = nn.Sequential(nn.Conv2d(out_dim*6, out_dim *2 , kernel_size=1),nn.BatchNorm2d(out_dim*2),act_fn,) # nn.Sequential(nn.Conv2d(out_dim*6, out_dim, kernel_size=1),nn.BatchNorm2d(out_dim),act_fn,)
        # 这里我们认为需要两个CA，一个给depth 一个给RGB 他们应该不一样，这样子能更好保留单模态specific的特征


        self.layer_ful2 = nn.Sequential(nn.Conv2d(out_dim , out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.gct = GCT(out_dim)

    def forward(self, rgb, depth):
        rgb = self.proj_r(rgb)
        depth = self.proj_d(depth)
        ################################
        n, c, h, w = rgb.shape

        m_r = self.avg(rgb)
        r_r = rgb - m_r
        v_r = self.linear_r(m_r)

        m_d = self.avg(depth)
        r_d = depth - m_d
        v_d = self.linear_d(m_d)
        
        

        v_mix = self.linear_m(v_r * v_d)
        alpha = self.cos(v_r[:,0,:], v_mix[:,0,:])
        beta = self.cos(v_d[:,0,:], v_mix[:,0,:])
        a_r = (alpha /(alpha + beta)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        b_d = (beta /(alpha + beta)).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        x_rgb = self.layer_10(r_r)
        x_dep = self.layer_20(r_d)
        
        
        avg_r = torch.mean(x_rgb,1).unsqueeze(1)
        avg_d = torch.mean(x_dep,1).unsqueeze(1)

        max_r = torch.max(x_rgb,1)[0].unsqueeze(1)
        max_d = torch.max(x_dep,1)[0].unsqueeze(1)

        att_r = self.layer_11(x_rgb)
        att_d = self.layer_21(x_dep)

        att = self.layer_att(torch.cat((att_r, att_d, max_r, max_d, avg_r, avg_r),dim=1))
        
        x_rgb_c = self.layer_30(r_r)
        x_dep_c = self.layer_40(r_d)
        
        avg_r_c = self.avg(x_rgb_c)
        avg_d_c = self.avg(x_dep_c)
        
        max_r_c = self.max(x_rgb_c)
        max_d_c = self.max(x_dep_c)
        
        gct_r_c = self.gct(x_rgb_c)
        gct_d_c = self.gct(x_dep_c)
             
        
        att_c = channel_shuffle (torch.cat([gct_r_c, gct_d_c ,avg_r_c,avg_d_c, max_r_c, max_d_c],dim=1),6)
        att_c = self.layer_att_c(att_c)
        att_cr, att_cd = att_c.chunk(2, dim=1) # 通过chunk操作可以把2*dim 的feature 变成 dim 和 dim
        

        att_r = att[:,0,:,:].unsqueeze(1).sigmoid()#?
        att_dep = att[:,1,:,:].unsqueeze(1).sigmoid()
        out = self.linear_f(torch.cat((rgb * a_r * att_r * att_cr , depth * b_d *att_dep* att_cd ), dim=1)).permute(0,2,1).view(rgb.shape)

        return out


class XMSNet(nn.Module):
    def __init__(self, pvt_pretrained=True, size_img=224):
        super(XMSNet, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample_32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
              
        
        self.pretrained = pvt_pretrained
        self.size_img = size_img

        # ---- PVT-V2 Backbone ----
        print(f"Creating model: {'pvt_v2_b5'}")
        self.pvtv2_depth = create_model('pvt_v2_b5',
            pretrained=self.pretrained,
            drop_rate=0.0,
            drop_path_rate=0.3,
            drop_block_rate=None,
            )
        self.pvtv2_rgb = create_model('pvt_v2_b5',
            pretrained=self.pretrained,
            drop_rate=0.0,
            drop_path_rate=0.3,
            drop_block_rate=None,
            )

        if self.pretrained:#pvt模型参数路径
            self.load_model('./checkpoints/pvt_v2_b5.pth') 
        
        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)

        ###############################################
        # funsion encoders #
        ###############################################
        
        
        self.fu_1 = AF0(64, 64) #MixedFusion_Block_IMfusion
        self.pool_fu_1 = maxpool()

        self.fu_2 = AF(128, 64,64)
        self.pool_fu_2 = maxpool()
        
        self.fu_3 = AF(320,64,64)
        self.pool_fu_3 = maxpool()

        self.fu_4 = AF(512,64,64)
        #self.pool_fu_4 = maxpool()
        
        
        ###############################################
        # decoders #
        ############################################### 
        self.lgb4 = LGB(64)
        self.lgb3 = LGB(64)
        self.lgb2 = LGB(64)
        self.lgb1 = LGB(64)
                              
        
        
        self.transformer4 = MaskAttentionTransformerBlock(64, 2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.transformer3 =  MaskAttentionTransformerBlock(64, 2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.transformer2 = MaskAttentionTransformerBlock(64, 2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.transformer1 =  MaskAttentionTransformerBlock(64, 2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
                
        
        self.gap = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1_1 = BasicConv2d(128,128,1,1,0)
       
        
        self.conv_128 = nn.Conv2d(256,128,1)
        
        self.t_mask3 =  nn.Sequential(BasicConv2d(64, 32, 3, padding=1), nn.Conv2d(32,1,1))
        self.t_mask2 =  nn.Sequential(BasicConv2d(64, 32, 3, padding=1), nn.Conv2d(32,1,1))
        self.t_mask1 = nn.Sequential(BasicConv2d(64, 32, 3, padding=1), nn.Conv2d(32,1,1))
        
        self.mask1 =  nn.Sequential(BasicConv2d(64, 32, 3, padding=1), nn.Conv2d(32,1,1))
        self.mask2 =  nn.Sequential(BasicConv2d(64, 32, 3, padding=1), nn.Conv2d(32,1,1))                    
        self.mask3 =  nn.Sequential(BasicConv2d(64, 32, 3, padding=1), nn.Conv2d(32,1,1))
        self.mask4 = nn.Sequential(BasicConv2d(64, 32, 3, padding=1), nn.Conv2d(32,1,1))                 
        #self.mask5 =  nn.Sequential(BasicConv2d(256, 128, 3, padding=1), nn.Conv2d(128,1,1))
        
        self.convr_4 =  nn.Sequential(BasicConv2d(512, 64, 3, padding=1))
        self.convd_4 =  nn.Sequential(BasicConv2d(512, 64, 3, padding=1))
        self.convr_3 =  nn.Sequential(BasicConv2d(320, 64, 3, padding=1))
        self.convd_3 =  nn.Sequential(BasicConv2d(320, 64, 3, padding=1))
        self.convr_2 =  nn.Sequential(BasicConv2d(128, 64, 3, padding=1))
        self.convd_2 =  nn.Sequential(BasicConv2d(128, 64, 3, padding=1))
        self.convr_1 =  nn.Sequential(BasicConv2d(64, 64, 3, padding=1))
        self.convd_1 =  nn.Sequential(BasicConv2d(64, 64, 3, padding=1))
        
        
        
        self.conv_cat4_3 =  nn.Sequential(BasicConv2d(64*2, 64, 3, padding=1))
        self.conv_cat3_2 =  nn.Sequential(BasicConv2d(64*2, 64, 3, padding=1))
        self.conv_cat2_1 =  nn.Sequential(BasicConv2d(64*2, 64, 3, padding=1))
        
        self.mlp4 = MLP_4d(input_dim=64, embed_dim=64)
        self.mlp3 = MLP_4d(input_dim=64, embed_dim=64)
        self.mlp2 = MLP_4d(input_dim=64, embed_dim=64)
        self.mlp1 = MLP_4d(input_dim=64, embed_dim=64)
        
        #加载预训练参数
    def load_model(self,pretrained):
        pretrained_dict = torch.load(pretrained)
        model_dict = self.pvtv2_rgb.state_dict()
        print("Load pretrained parameters from {}".format(pretrained))
        for k, v in pretrained_dict.items():
            #pdb.set_trace()
            if (k in model_dict):
                print("load:%s"%k)
            else:
                print("jump over:%s"%k)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)        
        #pdb.set_trace()
        self.pvtv2_depth.load_state_dict(model_dict)
        self.pvtv2_rgb.load_state_dict(model_dict)
        
        
              
    def forward(self, imgs, depths):
        
        img_1, img_2, img_3, img_4 = self.pvtv2_rgb(imgs)
        dep_1, dep_2, dep_3, dep_4 = self.pvtv2_depth(self.layer_dep0(depths))



        #lambdaLWA = self.depth_qaulity(img_1,dep_1)
        ####################################################
        ## fusion
        ####################################################
   
        ful_1    = self.fu_1(img_1, dep_1)#88 256 128
        ful_2    = self.fu_2(img_2, dep_2,  self.pool_fu_1(ful_1))# 44 512 256
        ful_3    = self.fu_3(img_3, dep_3,  self.pool_fu_2(ful_2))#22 1024 256  
        ful_4    = self.fu_4(img_4, dep_4,  self.pool_fu_3(ful_3))#11 2048 256

        ####################################################
        ## decoder fusion
        ###layer4
       
        lgb4 = self.lgb4(ful_4)#7 7 64
        g_ful_4 = self.gap(ful_4)
        dec_4 = self.mlp4(lgb4 * g_ful_4)
        mask4 = self.mask4(dec_4)
        map4  = self.upsample_32(mask4)
        transformer4 = self.transformer4( self.convr_4(img_4),self.convd_4(dep_4),dec_4*mask4.sigmoid())
        
        ###layer3
        lgb3 = self.lgb3(ful_3)#14 14 64
        dec_3 = self.mlp3(lgb3 * self.upsample_2(dec_4))
        mask3 = self.mask3(dec_3)
        map3  = self.upsample_16(mask3)
        transformer3 = self.transformer3( self.convr_3(img_3),self.convd_3(dep_3),dec_3*mask3.sigmoid())
        
        ###layer2
        lgb2 = self.lgb2(ful_2)#28 28 64
        dec_2 = self.mlp2(lgb2 * self.upsample_2(dec_3))
        mask2 = self.mask2(dec_2)
        map2  = self.upsample_8(mask2)
        transformer2 =self.transformer2( self.convr_2(img_2),self.convd_2(dep_2),dec_2*mask2.sigmoid())
        
        ###layer1           
        lgb1 = self.lgb1(ful_1)#56 56 64
        dec_1 = self.mlp1( lgb1 * self.upsample_2(dec_2))
        mask1 = self.mask1(dec_1)
        map1  = self.upsample_4(mask1)
        transformer1 = self.transformer1(self.convr_1(img_1),self.convd_1(dep_1),dec_1*mask1.sigmoid())
        
                
        ####################################################   
     ###########decoder1
        ###########
        
        conv_cat4_3 = self.conv_cat4_3(torch.cat([self.upsample_2(transformer4),transformer3],dim=1))#22
        conv_cat3_2 = self.conv_cat3_2(torch.cat([self.upsample_2(conv_cat4_3),transformer2],dim=1))#44
        conv_cat2_1 = self.conv_cat2_1(torch.cat([self.upsample_2(conv_cat3_2),transformer1],dim=1))#88
     ###########decoder2 
        t_mask3 = self.upsample_16(self.t_mask3(conv_cat4_3))
        t_mask2 = self.upsample_8(self.t_mask2(conv_cat3_2))
        t_mask1 = self.upsample_4(self.t_mask1(conv_cat2_1))
        
        
        
        
        
        ####################################################        

        


        return map4,map3,map2,map1,t_mask3, t_mask2,t_mask1


    


# Test code 

if __name__ == '__main__':
    rgb = torch.rand((2,3,352,352)).cuda()
    depth = torch.rand((2,1,352,352)).cuda()
    model = MRNet(pvt_pretrained=True).cuda()
    l = model(rgb,depth)
    print(l.size())
