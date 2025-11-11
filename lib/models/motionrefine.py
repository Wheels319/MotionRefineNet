import torch
from torch import Tensor, nn
from einops import rearrange
import pytorch_wavelets as tw

class MotionRefineNetTemBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, feature, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.linear5 = nn.Linear(in_channels//2, hidden_channels)
        self.linear6 = nn.Linear(hidden_channels, in_channels//2)
        self.linear7 = nn.Linear(in_channels, hidden_channels)
        self.linear8 = nn.Linear(hidden_channels, in_channels)
        self.linear9 = nn.Linear(in_channels//4, hidden_channels)
        self.linear10 = nn.Linear(hidden_channels, in_channels//4)
        self.linear11 = nn.Linear(in_channels//2, hidden_channels)
        self.linear12 = nn.Linear(hidden_channels, in_channels//2)
        self.linear3 = nn.Linear(feature, 512)
        self.linear4 = nn.Linear(512, feature)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.LN1 = nn.LayerNorm(feature)
        self.LN2 = nn.LayerNorm(in_channels)
        self.conv_layer = nn.Conv1d(in_channels=feature, out_channels=2*feature,kernel_size=1,stride=2,padding=0)
        self.conv_transpose_layer=nn.ConvTranspose1d(in_channels=2*feature,out_channels=feature,kernel_size=1,stride=2,padding=0,output_padding=1)
        self.conv_layer2 = nn.Conv1d(in_channels=2*feature, out_channels=4*feature, kernel_size=1, stride=2,
                                    padding=0)
        self.conv_transpose_layer2 = nn.ConvTranspose1d(in_channels=4*feature, out_channels=2*feature,
                                                       kernel_size=1, stride=2, padding=0,output_padding=1)
        self.conv_layer3 = nn.Conv1d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=1, stride=2,
                                    padding=0)
        self.conv_layer4 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=2,
                                    padding=0)

    def forward(self, x):
        x = rearrange(x, 'b s f -> b f s').contiguous()
        identity = x
        x = self.LN1(x)
        x = self.linear3(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear4(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x2 = x +identity
        x2 = rearrange(x2, 'b f s -> b s f').contiguous()

        identity2 = x2
        x2 = self.LN2(x2)
        x2 = self.linear1(x2)
        x2 = self.dropout(x2)
        x2 = self.lrelu(x2)
        x2 = self.linear2(x2)
        x2 = self.dropout(x2)
        x2 = self.lrelu(x2)

        x3 = self.conv_layer(x2)
        x3 = self.linear5(x3)
        x3 = self.dropout(x3)
        x3 = self.lrelu(x3)
        x3 = self.linear6(x3)
        x3 = self.dropout(x3)
        x3 = self.lrelu(x3)

        x4 = self.conv_layer2(x3)
        x4 = self.linear9(x4)
        x4 = self.dropout(x4)
        x4 = self.lrelu(x4)
        x4 = self.linear10(x4)
        x4 = self.dropout(x4)
        x4 = self.lrelu(x4)
        x4 = self.conv_transpose_layer2(x4)

        x5 = torch.cat((x3, x4), 1)
        x5 = self.linear11(x5)
        x5 = self.dropout(x5)
        x5 = self.lrelu(x5)
        x5 = self.linear12(x5)
        x5 = self.dropout(x5)
        x5 = self.lrelu(x5)
        x5 = rearrange(x5, 'b s f -> b f s').contiguous()
        x5 = self.conv_layer3(x5)
        x5 = rearrange(x5, 'b f s -> b s f').contiguous()
        x5 = self.conv_transpose_layer(x5)

        x6 = torch.cat((x2,x5), 1)
        x6 = self.linear7(x6)
        x6 = self.dropout(x6)
        x6 = self.lrelu(x6)
        x6 = self.linear8(x6)
        x6 = self.dropout(x6)
        x6 = self.lrelu(x6)
        x6 = rearrange(x6, 'b s f -> b f s').contiguous()
        x6 = self.conv_layer4(x6)
        x6 = rearrange(x6, 'b f s -> b s f').contiguous()

        out = x6 + identity2

        return out

class MotionRefineNetFreBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, feature, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.linear3 = nn.Linear(feature, 512)
        self.linear4 = nn.Linear(512, feature)
        self.linear5 = nn.Linear(in_channels//4, hidden_channels)
        self.linear6 = nn.Linear(hidden_channels, in_channels)
        self.linear7 = nn.Linear(in_channels//2, hidden_channels)
        self.linear8 = nn.Linear(in_channels//4, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.split_idx = 128
        self.LN1 = nn.LayerNorm(feature)
        self.LN2 = nn.LayerNorm(in_channels)
        self.afilter = nn.Sequential(
            nn.Linear(512, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 512)
        )

    def forward(self, x):
        group_h_i = torch.cat((x[:, :6, :], x[:, 12:24, :], x[:, 30:36, :]), 1)     # 3d
        # group_h_i = torch.cat((x2[:, :4, :], x2[:, 8:16, :], x2[:, 20:24, :]), 1)          # 2d

        x = rearrange(x, 'b s f -> b f s').contiguous()
        identity = x
        x = self.LN1(x)
        x = self.linear3(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear4(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x2 = x + identity
        x2 = rearrange(x2, 'b f s -> b s f').contiguous()

        identity2 = x2
        x2 = self.LN2(x2)
        # group_h = torch.cat((x2[:, :6, :], x2[:, 12:24, :], x2[:, 30:36, :]), 1)    # h36m 3dpw
        # group_l = torch.cat((x2[:, 6:12, :], x2[:, 24:30, :], x2[:, 36:51, :]), 1)

        group_h = torch.cat((x2[:, :6, :], x2[:, 12:24, :], x2[:, 30:36, :]), 1)         # aist
        group_l = torch.cat((x2[:, 6:12, :], x2[:, 24:30, :], x2[:, 36:42, :]), 1)

        #group_h = torch.cat((x2[:, :4, :], x2[:, 8:16, :], x2[:, 20:24, :]), 1)          # 2d
        #group_l = torch.cat((x2[:, 4:8, :], x2[:, 16:20, :], x2[:, 24:34, :]), 1)
        #group_l = torch.cat((x2[:, 4:8, :], x2[:, 16:20, :], x2[:, 24:32, :]), 1)

        mean = torch.mean(group_h_i, dim=2, keepdim=True)
        af=group_h/mean
        af = torch.mean(af, dim=1, keepdim=True)
        af = self.afilter(af)
        group_h = group_h * af

        group_h = self.linear1(group_h)
        group_h = self.dropout(group_h)
        group_h = self.lrelu(group_h)
        group_h = self.linear2(group_h)
        group_h = self.dropout(group_h)
        group_h = self.lrelu(group_h)

        x3_1 = self.linear5(group_l[:,:,:self.split_idx])
        x3_2 = self.linear7(group_l[:, :, self.split_idx:self.split_idx*3])
        x3_3 = self.linear8(group_l[:, :, self.split_idx*3:])
        group_l=x3_1*0.8+x3_2*0.1+x3_3*0.1
        group_l = self.dropout(group_l)
        group_l = self.lrelu(group_l)
        group_l = self.linear6(group_l)
        group_l = self.dropout(group_l)
        group_l = self.lrelu(group_l)

        # x3 = torch.cat((group_h[:, :6, :], group_l[:, :6, :], group_h[:, 6:18, :], group_l[:, 6:12, :],
        #                    group_h[:, 18:24, :], group_l[:, 12:27, :]), 1)       # h36m 3dpw

        x3 = torch.cat((group_h[:, :6, :], group_l[:, :6, :], group_h[:, 6:18, :], group_l[:, 6:12, :],
                           group_h[:, 18:24, :], group_l[:, 12:18, :]), 1)   # aist

        # x3 = torch.cat((group_h[:, :4, :], group_l[:, :4, :], group_h[:, 4:12, :], group_l[:, 4:8, :],
        #                    group_h[:, 12:16, :], group_l[:, 8:18, :]), 1)    # 2d 34

        # x3 = torch.cat((group_h[:, :4, :], group_l[:, :4, :], group_h[:, 4:12, :], group_l[:, 4:8, :],
        #                     group_h[:, 12:16, :], group_l[:, 8:16, :]), 1)   # 2d   32

        out = x3 + identity2

        return out

class MotionRefineNetFreBlock2(nn.Module):
    def __init__(self, in_channels, hidden_channels, feature, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.linear3 = nn.Linear(feature, 512)
        self.linear4 = nn.Linear(512, feature)
        self.linear5 = nn.Linear(in_channels//4, hidden_channels)
        self.linear6 = nn.Linear(hidden_channels, in_channels)
        self.linear7 = nn.Linear(in_channels//2, hidden_channels)
        self.linear8 = nn.Linear(in_channels//4, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.split_idx = 128
        self.LN1 = nn.LayerNorm(feature)
        self.LN2 = nn.LayerNorm(in_channels)
        self.afilter = nn.Sequential(
            nn.Linear(512, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 512)
        )

    def forward(self, x):
        x = rearrange(x, 'b s f -> b f s').contiguous()
        identity = x
        x = self.LN1(x)
        x = self.linear3(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear4(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x2 = x + identity
        x2 = rearrange(x2, 'b f s -> b s f').contiguous()

        identity2 = x2
        x2 = self.LN2(x2)
        # group_h = torch.cat((x2[:, :6, :], x2[:, 12:24, :], x2[:, 30:36, :]), 1)  # h36m 3dpw
        # group_l = torch.cat((x2[:, 6:12, :], x2[:, 24:30, :], x2[:, 36:51, :]), 1)

        group_h = torch.cat((x2[:, :6, :], x2[:, 12:24, :], x2[:, 30:36, :]), 1)         # aist
        group_l = torch.cat((x2[:, 6:12, :], x2[:, 24:30, :], x2[:, 36:42, :]), 1)

        # group_h = torch.cat((x2[:, :4, :], x2[:, 8:16, :], x2[:, 20:24, :]), 1)          # 2d
        # group_l = torch.cat((x2[:, 4:8, :], x2[:, 16:20, :], x2[:, 24:34, :]), 1)
        # group_l = torch.cat((x2[:, 4:8, :], x2[:, 16:20, :], x2[:, 24:32, :]), 1)

        af = torch.mean(group_h, dim=1, keepdim=True)
        af = self.afilter(af)
        group_h = group_h * af

        group_h = self.linear1(group_h)
        group_h = self.dropout(group_h)
        group_h = self.lrelu(group_h)
        group_h = self.linear2(group_h)
        group_h = self.dropout(group_h)
        group_h = self.lrelu(group_h)

        x3_1 = self.linear5(group_l[:, :, :self.split_idx])
        x3_2 = self.linear7(group_l[:, :, self.split_idx:self.split_idx * 3])
        x3_3 = self.linear8(group_l[:, :, self.split_idx * 3:])
        group_l = x3_1 * 0.8 + x3_2 * 0.1 + x3_3 * 0.1
        group_l = self.dropout(group_l)
        group_l = self.lrelu(group_l)
        group_l = self.linear6(group_l)
        group_l = self.dropout(group_l)
        group_l = self.lrelu(group_l)

        # x3 = torch.cat((group_h[:, :6, :], group_l[:, :6, :], group_h[:, 6:18, :], group_l[:, 6:12, :],
        #                     group_h[:, 18:24, :], group_l[:, 12:27, :]), 1)  # h36m 3dpw

        x3 = torch.cat((group_h[:, :6, :], group_l[:, :6, :], group_h[:, 6:18, :], group_l[:, 6:12, :],
                           group_h[:, 18:24, :], group_l[:, 12:18, :]), 1)   # aist

        # x3 = torch.cat((group_h[:, :4, :], group_l[:, :4, :], group_h[:, 4:12, :], group_l[:, 4:8, :],
        #                    group_h[:, 12:16, :], group_l[:, 8:18, :]), 1)    # 2d 34

        # x3 = torch.cat((group_h[:, :4, :], group_l[:, :4, :], group_h[:, 4:12, :], group_l[:, 4:8, :],
        #                     group_h[:, 12:16, :], group_l[:, 8:16, :]), 1)   # 2d   32

        out = x3 + identity2

        return out


class MotionRefineNet(nn.Module):
    def __init__(self,
                 window_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 res_hidden_size: int = 256,
                 num_blocks: int = 3,
                 dropout: float = 0.5,
                 split_idx: int = 128):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.res_hidden_size = res_hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.split_idx = split_idx
        joint_num = 42
        self.joint_num=joint_num

        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True))

        self.encoder2 = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True))

        tem_blocks = []
        for _ in range(num_blocks):
            tem_blocks.append(
                MotionRefineNetTemBlock(
                    in_channels=hidden_size,
                    hidden_channels=res_hidden_size,
                    feature=joint_num,
                    dropout=dropout))
        self.tem_blocks = nn.Sequential(*tem_blocks)

        self.fre_blocks = MotionRefineNetFreBlock(
                    in_channels=hidden_size,
                    hidden_channels=res_hidden_size,
                    feature=joint_num,
                    dropout=dropout)

        fre_blocks2 = []
        for _ in range(num_blocks-1):
            fre_blocks2.append(
                MotionRefineNetFreBlock2(
                    in_channels=hidden_size,
                    hidden_channels=res_hidden_size,
                    feature=joint_num,
                    dropout=dropout))
        self.fre_blocks2 = nn.Sequential(*fre_blocks2)

        self.decoder = nn.Linear(hidden_size, output_size)
        self.DWT = tw.DWT1D(J=2, wave='db1', mode='symmetric').cuda()
        self.IDWT = tw.IDWT1D(wave='db1', mode='symmetric').cuda()
        self.fusion=nn.Linear(hidden_size*2,2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        N, C, T = x.shape
        x = x.to(torch.float32)

        x_t = self.encoder(x)
        x_t = self.tem_blocks(x_t)

        x_f = self.encoder2(x)
        x_f_l, x_f_h = self.DWT(x_f)
        x_f = torch.cat((x_f_l, x_f_h[0], x_f_h[1]), 2)
        x_f = self.fre_blocks(x_f)
        x_f = self.fre_blocks2(x_f)
        x_f_l = x_f[:, :, :self.split_idx]
        x_f_h = []
        x_f_h.append(x_f[:, :, self.split_idx:3 * self.split_idx])
        x_f_h.append(x_f[:, :, 3 * self.split_idx:])
        x_f = self.IDWT((x_f_l, x_f_h))

        out = self.decoder((x_t + x_f) / 2)

        return out
