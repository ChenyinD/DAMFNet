import torch
import torch.nn as nn
from .backbone import build_backbone
from .modules import TransformerDecoder, Transformer
from einops import rearrange


class token_encoder(nn.Module):
    def __init__(self, in_chan = 32, token_len = 4, heads = 4):
        super(token_encoder, self).__init__()
        self.token_len = token_len
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len, in_chan))
        self.transformer = Transformer(dim=in_chan, depth=1, heads=heads, dim_head=64, mlp_dim=64, dropout=0)

    def forward(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()

        tokens = torch.einsum('bln, bcn->blc', spatial_attention, x)

        tokens += self.pos_embedding
        x = self.transformer(tokens)
        return x

class token_decoder(nn.Module):
    def __init__(self, in_chan = 32, size = 32, heads = 4):
        super(token_decoder, self).__init__()
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, in_chan, size, size))
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0,softmax=in_chan)

    def forward(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x


class context_aggregator(nn.Module):
    def __init__(self, in_chan=32, size=32):
        super(context_aggregator, self).__init__()
        self.token_encoder = token_encoder(in_chan=in_chan, token_len=4)
        self.token_decoder = token_decoder(in_chan = in_chan, size = size, heads = 8)
        # self.con=nn.Conv2d(in_chan,in_chan//2,1,1,0)

    def forward(self, feature1,feature2):
        # feature=torch.cat((feature1,feature2),dim=1)
        token1 = self.token_encoder(feature1)
        out1 = self.token_decoder(feature1, token1)
        token2 = self.token_encoder(feature2)
        out2 = self.token_decoder(feature2, token2)
        out=torch.abs(out2-out1)
        # out=self.con(out)
        return out
class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,groups=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class SW(nn.Module):
    def __init__(self):
        super(SW, self).__init__()
        self.conv =  nn.Sequential(
            nn.Conv2d(2, 2, 3,1,1),
            nn.Conv2d(2, 2, 3, 1, 1),
            nn.Conv2d(2, 2, 3, 1, 1),
            nn.Conv2d(2, 1, 1, 1, 0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y*x
        return y

class CN(nn.Module):
    def __init__(self,chan):
        super(CN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(chan, chan//16, 1,bias=False),
            # nn.InstanceNorm2d(chan//16),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan//16 ,chan, 1,bias=False),
            # nn.InstanceNorm2d(chan),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
            avgout = self.shared_MLP(self.avg_pool(x))
            maxout = self.shared_MLP(self.max_pool(x))
            y=self.sigmoid(avgout + maxout)
            y=y*x
            return y
class CAT(nn.Module):
    def __init__(self,chan):
        super(CAT,self).__init__()
        self.cn=CN(chan)
        self.sw=SW()
        self.con1=nn.Conv2d(chan*2, chan, 1, 1, 0)
        self.con3=nn.Sequential(
            nn.Conv2d(chan,chan,3,1,1),
            nn.InstanceNorm2d(chan),
            nn.ReLU(inplace=True),


        )

    def forward(self, dif):

        dcn3=self.cn(dif)
        dcn3=dcn3+dif
        img3=self.sw(dcn3)
        img3=img3+dif
        out=self.con3(img3)
        # all=self.con1(all)
        return out

class BGA(nn.Module):
    def __init__(self,inchnnel,outchnnel,):
        super(BGA, self).__init__()
        self.smself=nn.Sequential(
            depthwise_separable_conv(inchnnel,outchnnel),
            nn.InstanceNorm2d(outchnnel)
        )
        self.lagself=nn.Sequential(
            depthwise_separable_conv(outchnnel,outchnnel),
            nn.InstanceNorm2d(outchnnel)
        )
        self.smout=nn.Sequential(
            nn.Conv2d(inchnnel,inchnnel,3,1,1),
            nn.InstanceNorm2d(inchnnel),
            nn.Conv2d(inchnnel,outchnnel*2,1,1),
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(outchnnel*2, outchnnel, 1, 1),
            nn.Sigmoid()
        )
        self.lagout=nn.Sequential(
            nn.Conv2d(outchnnel, outchnnel, 3, 1, 1),
            nn.InstanceNorm2d(outchnnel),
            nn.AvgPool2d(3,2,1),
            nn.Sigmoid()
        )
        self.up= nn.Sequential(
            nn.Conv2d(outchnnel,outchnnel*2,1,1),
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(outchnnel*2, outchnnel, 1, 1),
        )
        self.ou=nn.Sequential(
            nn.Conv2d(outchnnel, outchnnel, 3, 1, 1),
            nn.InstanceNorm2d(outchnnel),
        )

    def forward(self, L, S):
        Lself=self.lagself(L)
        Lout=self.lagout(L)
        Sself=self.smself(S)
        Sout=self.smout(S)
        Lself=Sout*Lself
        Sself=Sself*Lout
        Sself=self.up(Sself)
        out=Sself+Lself
        out=self.ou(out)
        return out
class BGA2(nn.Module):
    def __init__(self,inchnnel,outchnnel,):
        super(BGA2, self).__init__()
        self.smself=nn.Sequential(
            depthwise_separable_conv(inchnnel,outchnnel),
            nn.InstanceNorm2d(outchnnel)
        )
        self.lagself=nn.Sequential(
            depthwise_separable_conv(outchnnel,outchnnel),
            nn.InstanceNorm2d(outchnnel)
        )
        self.smout=nn.Sequential(
            nn.Conv2d(inchnnel,inchnnel,3,1,1),
            nn.InstanceNorm2d(inchnnel),
            nn.Conv2d(inchnnel,outchnnel,1,1),
            nn.Sigmoid()
        )
        self.lagout=nn.Sequential(
            nn.Conv2d(outchnnel, outchnnel, 3, 1, 1),
            nn.InstanceNorm2d(outchnnel),
            nn.Sigmoid()
        )
        self.ou=nn.Sequential(
            nn.Conv2d(outchnnel, outchnnel, 3, 1, 1),
            nn.InstanceNorm2d(outchnnel),
        )

    def forward(self, L, S):
        Lself=self.lagself(L)
        Lout=self.lagout(L)
        Sself=self.smself(S)
        Sout=self.smout(S)
        Lself=Sout*Lself
        Sself=Sself*Lout
        out=Sself+Lself
        out=self.ou(out)
        return out
class PPM(nn.Module):
    def __init__(self):
        super(PPM,self).__init__()
        self.avg1=nn.AdaptiveAvgPool2d(1)
        self.avg2 = nn.AdaptiveAvgPool2d(2)
        self.avg4 = nn.AdaptiveAvgPool2d(4)
        self.avg6 = nn.AdaptiveAvgPool2d(8)
        self.con=nn.Conv2d(512,128,1,1,0)
        self.up = nn.Upsample(16, mode="bilinear", align_corners=True)
        # self.con1 = nn.Sequential(
        #     nn.Conv2d(512, 256, 1, 1, 0),
        #     nn.Sigmoid()
        # )
    def forward(self, out):
        out1 = self.up(self.con(self.avg1(out)))
        out2 = self.up(self.con(self.avg2(out)))
        out4 = self.up(self.con(self.avg4(out)))
        out6 = self.up(self.con(self.avg6(out)))
        out_all = torch.cat((out1, out2, out4, out6), dim=1)
        # out_all=self.con1(out_all)
        return out_all
class PRD(nn.Module):
    def __init__(self):
        super(PRD,self).__init__()
        self.con0=nn.Conv2d(64,64,1,1)
        self.con1=nn.Conv2d(64,64,3,1,1,1)
        self.con2=nn.Conv2d(64,64,3,1,2,2)
        self.con3=nn.Conv2d(64,64,3,1,3,3)
        self.con4 = nn.Conv2d(64, 64, 3, 1, 4, 4)

    def forward(self, out):
        c0=self.con0(out)
        c1=self.con1(out)
        c2=self.con2(out)
        c3=self.con3(out)
        c4=self.con4(out)
        c=c0+c1+c2+c3+c4
        return c


class CDNet(nn.Module):
    def __init__(self,backbone='resnet18', output_stride=16,img_chan=3,):
        super(CDNet, self).__init__()
        BatchNorm = nn.InstanceNorm2d
        self.backbone=build_backbone(backbone, output_stride, BatchNorm, img_chan)
        self.up1=nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(64,1,1,1,0)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bicubic", align_corners=True),
            nn.Conv2d(64,1,1,1,0)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="bicubic", align_corners=True),
            nn.Conv2d(128,1,1,1,0)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode="bicubic", align_corners=True),
            nn.Conv2d(256,1,1,1,0)
        )
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode="bicubic", align_corners=True),
            nn.Conv2d(512, 1, 1, 1, 0)
        )
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bicubic", align_corners=True),
            nn.Conv2d(128, 1, 1, 1, 0)
        )
        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="bicubic", align_corners=True),
            nn.Conv2d(256, 1, 1, 1, 0)
        )
        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode="bicubic", align_corners=True),
            nn.Conv2d(512, 1, 1, 1, 0)
        )
        self.up9 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode="bicubic", align_corners=True),
            nn.Conv2d(1024, 1, 1, 1, 0)
        )
        self.con1=nn.Conv2d(128,64,1,bias=False)
        self.con2 = nn.Conv2d(256, 128, 1, bias=False)
        self.con3 = nn.Conv2d(512, 256, 1, bias=False)
        self.con4 = nn.Conv2d(1024, 512, 1, bias=False)
        self.con5 = nn.Conv2d(1024, 1, 1, bias=False)
        self.tran1 = context_aggregator(in_chan=64, size=64)
        self.tran2 = context_aggregator(in_chan=128, size=32)
        self.tran3 = context_aggregator(in_chan=256, size=16)
        self.tran4 = context_aggregator(in_chan=512, size=16)
        self.bga1 = BGA(128, 64)
        self.bga2 = BGA(256, 128)
        self.bga3 = BGA2(512, 256)
        self.out = nn.Sequential(
            nn.Conv2d(64, 512, 1, 1),
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(512, 64, 1, 1),
            nn.InstanceNorm2d(64),
            PRD(),
            nn.Conv2d(64, 512, 1, 1),
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True),
            nn.Conv2d(512, 64, 1, 1),
            nn.InstanceNorm2d(64),
            PRD(),
            nn.Conv2d(64, 1, 1, 1)
        )
        self.ppm = PPM()
        self.afm1 = CAT(64)
        self.afm2 = CAT(128)
        self.afm3 = CAT(256)
        self.afm4 = CAT(512)


    def forward(self, img1,img2):
        x1,x2,x3,x4 = self.backbone(img1)
        y1,y2,y3,y4= self.backbone(img2)
        d1 = self.afm1(torch.abs(y1-x1))
        d2 = self.afm2(torch.abs(y2-x2))
        d3 = self.afm3(torch.abs(y3-x3))
        d4 = self.afm4(torch.abs(y4-x4))
        t1 = self.tran1(x1,y1)
        t2 = self.tran2(x2, y2)
        t3 = self.tran3(x3, y3)
        t4 = self.tran4(x4, y4)
        all1= self.con1(torch.cat((t1,d1),dim=1))
        all2 = self.con2(torch.cat((t2, d2), dim=1))
        all3 = self.con3(torch.cat((t3, d3), dim=1))
        all4 = self.con4(torch.cat((t4, d4), dim=1))
        out1 = self.up2(all1)
        out2 = self.up3(all2)
        out3 = self.up4(all3)
        out4 = self.up5(all4)
        ppm4 = self.ppm(all4)
        all4 = all4+ppm4
        all3 = self.bga3(all3,all4)
        all2 = self.bga2(all2, all3)
        all1 = self.bga1(all1, all2)
        out = self.out(all1)

        return out,out1,out2,out3,out4
