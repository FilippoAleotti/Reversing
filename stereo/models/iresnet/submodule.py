import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def conv2d(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=True,
        )
    )


def image_warp(img, depth, padding_mode="zeros"):

    # img: the source image (where to sample pixels) -- [B, 3, H, W]
    # depth: depth map of the target image -- [B, 1, H, W]
    # Returns: Source image warped to the target image

    b, _, h, w = depth.size()
    i_range = torch.autograd.Variable(
        torch.linspace(0, h - 1, steps=h).view(1, h, 1).expand(1, h, w),
        requires_grad=False,
    )  # [1, H, W]  copy 0-height for w times : y coord
    j_range = torch.autograd.Variable(
        torch.linspace(0, w - 1, steps=w).view(1, 1, w).expand(1, h, w),
        requires_grad=False,
    )  # [1, H, W]  copy 0-width for h times  : x coord

    pixel_coords = torch.stack((j_range, i_range), dim=1).float().cuda()  # [1, 2, H, W]
    batch_pixel_coords = (
        pixel_coords[:, :, :, :].expand(b, 2, h, w).contiguous().view(b, 2, -1)
    )  # [B, 2, H*W]

    X = batch_pixel_coords[:, 0, :] - depth.contiguous().view(b, -1)  # [B, H*W]
    Y = batch_pixel_coords[:, 1, :]

    X_norm = X * 2 / (w - 1) - 1
    Y_norm = Y * 2 / (h - 1) - 1

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    pixel_coords = pixel_coords.view(b, h, w, 2)  # [B, H, W, 2]

    projected_img = torch.nn.functional.grid_sample(
        img, pixel_coords, padding_mode=padding_mode
    )

    return projected_img


class stem_block(nn.Module):
    def __init__(self):
        super(stem_block, self).__init__()

        self.conv1 = nn.Sequential(conv2d(3, 64, 7, 2, 3, 1), nn.ReLU(inplace=True))

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            ),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(conv2d(64, 128, 5, 2, 2, 1), nn.ReLU(inplace=True))

        self.up_2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 32, kernel_size=8, padding=2, output_padding=0, stride=4, bias=True
            ),
            nn.ReLU(inplace=True),
        )

        self.up_12 = nn.Sequential(conv2d(64, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))

    def forward(self, x):

        batch_size = x.size()[0]
        height = x.size()[2]
        width = x.size()[3]

        conv1 = self.conv1(x)
        up_1 = self.up_1(conv1)
        conv2 = self.conv2(conv1)
        up_2 = self.up_2(conv2)

        # up_1+up_2
        concat = Variable(
            torch.FloatTensor(batch_size, 64, height, width).zero_()
        ).cuda()
        concat[:, :32, :, :] = up_1[:, :, :, :]
        concat[:, 32:, :, :] = up_2[:, :, :, :]
        concat = concat.contiguous()

        up_12 = self.up_12(concat)

        return conv1, conv2, up_12


class disparity_estimation(nn.Module):
    def __init__(self):
        super(disparity_estimation, self).__init__()

        self.conv_redir = nn.Sequential(
            conv2d(128, 64, 1, 1, 0, 1), nn.ReLU(inplace=True)
        )

        self.conv3_1 = nn.Sequential(
            conv2d(145, 256, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            conv2d(256, 256, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.conv4_1 = nn.Sequential(
            conv2d(256, 512, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            conv2d(512, 512, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv5_1 = nn.Sequential(
            conv2d(512, 512, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            conv2d(512, 512, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv6_1 = nn.Sequential(
            conv2d(512, 1024, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            conv2d(1024, 1024, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.disp6 = nn.Sequential(conv2d(1024, 1, 3, 1, 1, 1))
        self.udisp6 = nn.Sequential(
            nn.ConvTranspose2d(
                1, 1, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            )
        )

        self.uconv5 = nn.Sequential(
            nn.ConvTranspose2d(
                1024,
                512,
                kernel_size=4,
                padding=1,
                output_padding=0,
                stride=2,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.iconv5 = nn.Sequential(
            conv2d(1025, 512, 3, 1, 1, 1), nn.ReLU(inplace=True)
        )

        self.disp5 = nn.Sequential(conv2d(512, 1, 3, 1, 1, 1))
        self.udisp5 = nn.Sequential(
            nn.ConvTranspose2d(
                1, 1, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            )
        )

        self.uconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                padding=1,
                output_padding=0,
                stride=2,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.iconv4 = nn.Sequential(conv2d(769, 256, 3, 1, 1, 1), nn.ReLU(inplace=True))

        self.disp4 = nn.Sequential(conv2d(256, 1, 3, 1, 1, 1))
        self.udisp4 = nn.Sequential(
            nn.ConvTranspose2d(
                1, 1, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            )
        )

        self.uconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                padding=1,
                output_padding=0,
                stride=2,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.iconv3 = nn.Sequential(conv2d(385, 128, 3, 1, 1, 1), nn.ReLU(inplace=True))

        self.disp3 = nn.Sequential(conv2d(128, 1, 3, 1, 1, 1))
        self.udisp3 = nn.Sequential(
            nn.ConvTranspose2d(
                1, 1, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            )
        )

        self.uconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            ),
            nn.ReLU(inplace=True),
        )

        self.iconv2 = nn.Sequential(conv2d(193, 64, 3, 1, 1, 1), nn.ReLU(inplace=True))

        self.disp2 = nn.Sequential(conv2d(64, 1, 3, 1, 1, 1))
        self.udisp2 = nn.Sequential(
            nn.ConvTranspose2d(
                1, 1, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            )
        )

        self.uconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            ),
            nn.ReLU(inplace=True),
        )

        self.iconv1 = nn.Sequential(conv2d(97, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))

        self.disp1 = nn.Sequential(conv2d(32, 1, 3, 1, 1, 1))
        self.udisp1 = nn.Sequential(
            nn.ConvTranspose2d(
                1, 1, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            )
        )

        self.uconv0 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 32, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            ),
            nn.ReLU(inplace=True),
        )

        self.iconv0 = nn.Sequential(conv2d(65, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))

        self.disp0 = nn.Sequential(conv2d(32, 1, 3, 1, 1, 1))

    def forward(self, conv1a, up_1a2a, conv2a, corr1d):

        batch_size = conv2a.size()[0]
        height = conv2a.size()[2]
        width = conv2a.size()[3]

        conv_redir = self.conv_redir(conv2a)

        corr1d_conv_redir = Variable(
            torch.FloatTensor(batch_size, corr1d.size()[1] + 64, height, width).zero_()
        ).cuda()
        corr1d_conv_redir[:, : corr1d.size()[1], :, :] = corr1d[:, :, :, :]
        corr1d_conv_redir[:, corr1d.size()[1] :, :, :] = conv_redir[:, :, :, :]
        corr1d_conv_redir = corr1d_conv_redir.contiguous()

        conv3_1 = self.conv3_1(corr1d_conv_redir)
        conv4_1 = self.conv4_1(conv3_1)
        conv5_1 = self.conv5_1(conv4_1)
        conv6_1 = self.conv6_1(conv5_1)

        disp6 = self.disp6(conv6_1)
        udisp6 = self.udisp6(disp6)

        uconv5 = self.uconv5(conv6_1)
        uconv5_disp6_conv5_1 = Variable(
            torch.FloatTensor(
                batch_size,
                uconv5.size()[1] + 1 + conv5_1.size()[1],
                height // 8,
                width // 8,
            ).zero_()
        ).cuda()
        uconv5_disp6_conv5_1[:, : uconv5.size()[1], :, :] = uconv5[:, :, :, :]
        uconv5_disp6_conv5_1[:, uconv5.size()[1] : uconv5.size()[1] + 1, :, :] = udisp6[
            :, :, :, :
        ]
        uconv5_disp6_conv5_1[:, uconv5.size()[1] + 1 :, :, :] = conv5_1[:, :, :, :]
        uconv5_disp6_conv5_1 = uconv5_disp6_conv5_1.contiguous()
        iconv5 = self.iconv5(uconv5_disp6_conv5_1)

        disp5 = self.disp5(iconv5)
        udisp5 = self.udisp5(disp5)

        uconv4 = self.uconv4(iconv5)
        uconv4_disp5_conv4_1 = Variable(
            torch.FloatTensor(
                batch_size,
                uconv4.size()[1] + 1 + conv4_1.size()[1],
                height // 4,
                width // 4,
            ).zero_()
        ).cuda()
        uconv4_disp5_conv4_1[:, : uconv4.size()[1], :, :] = uconv4[:, :, :, :]
        uconv4_disp5_conv4_1[:, uconv4.size()[1] : uconv4.size()[1] + 1, :, :] = udisp5[
            :, :, :, :
        ]
        uconv4_disp5_conv4_1[:, uconv4.size()[1] + 1 :, :, :] = conv4_1[:, :, :, :]
        uconv4_disp5_conv4_1 = uconv4_disp5_conv4_1.contiguous()
        iconv4 = self.iconv4(uconv4_disp5_conv4_1)

        disp4 = self.disp4(iconv4)
        udisp4 = self.udisp4(disp4)

        uconv3 = self.uconv3(iconv4)
        uconv3_disp4_conv3_1 = Variable(
            torch.FloatTensor(
                batch_size,
                uconv3.size()[1] + 1 + conv3_1.size()[1],
                height // 2,
                width // 2,
            ).zero_()
        ).cuda()
        uconv3_disp4_conv3_1[:, : uconv3.size()[1], :, :] = uconv3[:, :, :, :]
        uconv3_disp4_conv3_1[:, uconv3.size()[1] : uconv3.size()[1] + 1, :, :] = udisp4[
            :, :, :, :
        ]
        uconv3_disp4_conv3_1[:, uconv3.size()[1] + 1 :, :, :] = conv3_1[:, :, :, :]
        uconv3_disp4_conv3_1 = uconv3_disp4_conv3_1.contiguous()
        iconv3 = self.iconv3(uconv3_disp4_conv3_1)

        disp3 = self.disp3(iconv3)
        udisp3 = self.udisp3(disp3)

        uconv2 = self.uconv2(iconv3)
        uconv2_disp3_conv2a = Variable(
            torch.FloatTensor(
                batch_size, uconv2.size()[1] + 1 + conv2a.size()[1], height, width
            ).zero_()
        ).cuda()
        uconv2_disp3_conv2a[:, : uconv2.size()[1], :, :] = uconv2[:, :, :, :]
        uconv2_disp3_conv2a[:, uconv2.size()[1] : uconv2.size()[1] + 1, :, :] = udisp3[
            :, :, :, :
        ]
        uconv2_disp3_conv2a[:, uconv2.size()[1] + 1 :, :, :] = conv2a[:, :, :, :]
        uconv2_disp3_conv2a = uconv2_disp3_conv2a.contiguous()
        iconv2 = self.iconv2(uconv2_disp3_conv2a)

        disp2 = self.disp2(iconv2)
        udisp2 = self.udisp2(disp2)

        uconv1 = self.uconv1(iconv2)
        uconv1_disp2_conv1a = Variable(
            torch.FloatTensor(
                batch_size,
                uconv1.size()[1] + 1 + conv1a.size()[1],
                height * 2,
                width * 2,
            ).zero_()
        ).cuda()
        uconv1_disp2_conv1a[:, : uconv1.size()[1], :, :] = uconv1[:, :, :, :]
        uconv1_disp2_conv1a[:, uconv1.size()[1] : uconv1.size()[1] + 1, :, :] = udisp2[
            :, :, :, :
        ]
        uconv1_disp2_conv1a[:, uconv1.size()[1] + 1 :, :, :] = conv1a[:, :, :, :]
        uconv1_disp2_conv1a = uconv1_disp2_conv1a.contiguous()
        iconv1 = self.iconv1(uconv1_disp2_conv1a)

        disp1 = self.disp1(iconv1)
        udisp1 = self.udisp1(disp1)

        uconv0 = self.uconv0(iconv1)
        uconv0_disp1_up_1a2a = Variable(
            torch.FloatTensor(
                batch_size,
                uconv0.size()[1] + 1 + up_1a2a.size()[1],
                height * 4,
                width * 4,
            ).zero_()
        ).cuda()
        uconv0_disp1_up_1a2a[:, : uconv0.size()[1], :, :] = uconv0[:, :, :, :]
        uconv0_disp1_up_1a2a[:, uconv0.size()[1] : uconv0.size()[1] + 1, :, :] = udisp1[
            :, :, :, :
        ]
        uconv0_disp1_up_1a2a[:, uconv0.size()[1] + 1 :, :, :] = up_1a2a[:, :, :, :]
        uconv0_disp1_up_1a2a = uconv0_disp1_up_1a2a.contiguous()
        iconv0 = self.iconv0(uconv0_disp1_up_1a2a)

        disp0 = self.disp0(iconv0)

        return disp0, disp1, disp2, disp3, disp4, disp5, disp6


class disparity_refinement(nn.Module):
    def __init__(self):
        super(disparity_refinement, self).__init__()

        self.r_conv0 = nn.Sequential(conv2d(65, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))

        self.r_conv1 = nn.Sequential(conv2d(32, 64, 3, 2, 1, 1), nn.ReLU(inplace=True))

        self.c_conv1 = nn.Sequential(conv2d(64, 16, 3, 1, 1, 1), nn.ReLU(inplace=True))

        self.r_conv1_1 = nn.Sequential(
            conv2d(105, 64, 3, 1, 1, 1), nn.ReLU(inplace=True)
        )

        self.r_conv2 = nn.Sequential(conv2d(64, 128, 3, 2, 1, 1), nn.ReLU(inplace=True))

        self.r_conv2_1 = nn.Sequential(
            conv2d(128, 128, 3, 1, 1, 1), nn.ReLU(inplace=True)
        )

        self.r_res2 = nn.Sequential(conv2d(128, 1, 3, 1, 1, 1))
        self.r_res2_up = nn.Sequential(
            nn.ConvTranspose2d(
                1, 1, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            )
        )

        self.r_uconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            ),
            nn.ReLU(inplace=True),
        )

        self.r_iconv1 = nn.Sequential(
            conv2d(129, 64, 3, 1, 1, 1), nn.ReLU(inplace=True)
        )

        self.r_res1 = nn.Sequential(conv2d(64, 1, 3, 1, 1, 1))
        self.r_res1_up = nn.Sequential(
            nn.ConvTranspose2d(
                1, 1, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            )
        )

        self.r_uconv0 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, padding=1, output_padding=0, stride=2, bias=True
            ),
            nn.ReLU(inplace=True),
        )

        self.r_iconv0 = nn.Sequential(conv2d(65, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))

        self.r_res0 = nn.Sequential(conv2d(32, 1, 3, 1, 1, 1))

    def forward(self, up_1a2a, up_1b2b, conv1a, conv1b, disp0):

        batch_size = up_1a2a.size()[0]
        height = up_1a2a.size()[2]
        width = up_1a2a.size()[3]
        num_disp = 20

        # Warping
        w_up_1b2b = image_warp(up_1b2b, disp0)
        warp_error = torch.abs(up_1a2a - w_up_1b2b)

        volume = Variable(
            torch.FloatTensor(
                batch_size, warp_error.size()[1] + 1 + up_1a2a.size()[1], height, width
            ).zero_()
        ).cuda()
        volume[:, : warp_error.size()[1], :, :] = warp_error[:, :, :, :]
        volume[:, warp_error.size()[1] : warp_error.size()[1] + 1, :, :] = disp0[
            :, :, :, :
        ]
        volume[:, warp_error.size()[1] + 1 :, :, :] = up_1a2a[:, :, :, :]

        r_conv0 = self.r_conv0(volume)
        r_conv1 = self.r_conv1(r_conv0)

        c_conv1a = self.c_conv1(conv1a)
        c_conv1b = self.c_conv1(conv1b)

        # correlation layer
        r_corr = Variable(
            torch.FloatTensor(
                batch_size, num_disp * 2 + 1, height // 2, width // 2
            ).zero_()
        ).cuda()
        pad_conv1b = F.pad(c_conv1b, (num_disp, num_disp), "constant", 0)

        for i in range(-num_disp, num_disp + 1):
            if i != 0:
                r_corr[:, i + num_disp, :, :] = torch.mean(
                    c_conv1a[:, :, :, :]
                    * pad_conv1b[:, :, :, i + num_disp : i + num_disp + width // 2],
                    dim=1,
                    keepdim=False,
                )
            else:
                r_corr[:, num_disp, :, :] = torch.mean(
                    c_conv1a[:, :, :, :]
                    * pad_conv1b[:, :, :, num_disp : num_disp + width // 2],
                    dim=1,
                    keepdim=False,
                )
        r_corr = r_corr.contiguous()

        # concat r_conv1 + r_corr
        r_conv1_r_corr = Variable(
            torch.FloatTensor(
                batch_size, r_corr.size()[1] + 64, height // 2, width // 2
            ).zero_()
        ).cuda()
        r_conv1_r_corr[:, : r_conv1.size()[1], :, :] = r_conv1[:, :, :, :]
        r_conv1_r_corr[:, r_conv1.size()[1] :, :, :] = r_corr[:, :, :, :]
        r_conv1_r_corr = r_conv1_r_corr.contiguous()

        r_conv1_1 = self.r_conv1_1(r_conv1_r_corr)
        r_conv2 = self.r_conv2(r_conv1_1)
        r_conv2_1 = self.r_conv2_1(r_conv2)
        r_res2 = self.r_res2(r_conv2_1)
        r_res2_up = self.r_res2_up(r_res2)

        r_uconv1 = self.r_uconv1(r_conv2_1)

        # concat r_uconv1 + r_res2 + r_conv1_1
        r_uconv1_r_res2_up_r_conv1_1 = Variable(
            torch.FloatTensor(
                batch_size,
                r_uconv1.size()[1] + 1 + r_conv1_1.size()[1],
                height // 2,
                width // 2,
            ).zero_()
        ).cuda()
        r_uconv1_r_res2_up_r_conv1_1[:, : r_uconv1.size()[1], :, :] = r_uconv1[
            :, :, :, :
        ]
        r_uconv1_r_res2_up_r_conv1_1[
            :, r_uconv1.size()[1] : r_uconv1.size()[1] + 1, :, :
        ] = r_res2_up[:, :, :, :]
        r_uconv1_r_res2_up_r_conv1_1[:, r_uconv1.size()[1] + 1 :, :, :] = r_conv1_1[
            :, :, :, :
        ]
        r_uconv1_r_res2_up_r_conv1_1 = r_uconv1_r_res2_up_r_conv1_1.contiguous()

        r_iconv1 = self.r_iconv1(r_uconv1_r_res2_up_r_conv1_1)
        r_res1 = self.r_res1(r_iconv1)
        r_res1_up = self.r_res1_up(r_res1)

        r_uconv0 = self.r_uconv0(r_iconv1)

        # concat r_uconv0 + r_res1 + r_conv0
        r_uconv0_r_res1_up_r_conv1_1 = Variable(
            torch.FloatTensor(
                batch_size, r_uconv0.size()[1] + 1 + r_conv0.size()[1], height, width
            ).zero_()
        ).cuda()
        r_uconv0_r_res1_up_r_conv1_1[:, : r_uconv0.size()[1], :, :] = r_uconv0[
            :, :, :, :
        ]
        r_uconv0_r_res1_up_r_conv1_1[
            :, r_uconv0.size()[1] : r_uconv0.size()[1] + 1, :, :
        ] = r_res1_up[:, :, :, :]
        r_uconv0_r_res1_up_r_conv1_1[:, r_uconv0.size()[1] + 1 :, :, :] = r_conv0[
            :, :, :, :
        ]
        r_uconv0_r_res1_up_r_conv1_1 = r_uconv0_r_res1_up_r_conv1_1.contiguous()

        r_iconv0 = self.r_iconv0(r_uconv0_r_res1_up_r_conv1_1)
        r_res0 = self.r_res0(r_iconv0)

        return r_res0, r_res1, r_res2
