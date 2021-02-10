import torch
import torch.nn as nn

# This architecture is based exatly on the description provided in
# the section 2 "Network Architecture" of the U-Net paper.


def double_convolution(in_ch, out_ch, mid_ch=None, optimised=False):
    """The architecture consists of the repeated application of two 3x3 unpadded covolutions.
    Each convolution is followed by a ReLU.
    inplace=True because there is no need to create and store a copy
    mid_ch is used in the upwards path.

    Args:
        in_ch ([int]): [Number of input channels.]
        out_ch ([int]): [Number of output channels.]
        mid_ch ([int]): [Number of channels in the middle part of the up-convolutions.]
        optimised ([bool]): [Decides whether BatchNormalisation is applied or not.]

    Returns:
        [convolution]: [The callableobject for applying the convolution.]
    """
    if not mid_ch:
        mid_ch = out_ch
    convolution = nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, kernel_size=3),
        if optimised:
            nn.BatchNorm2d(mid_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, kernel_size=3),
        if optimised:
            nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
    return convolution


def up_transpose(in_ch, out_ch):
    """Up transpose operation for increasing the resolution of the image and
    decreasing the depth of the channels.
    Chose nn.ConvTranspose over nn.Upsample for higher accuracy due to learnable param.

    Args:
        in_ch ([int]): [Number of input channels.]
        out_ch ([type]): [Number of output channels.]

    Returns:
        [transpose]: [The modified image tensor.]
    """
    transpose = nn.ConvTranspose2d(
        in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2
    )
    return transpose


def crop_img(input_tensor, target_tensor):
    """Crop the image tensor from the encoder path to the correct size, which
    is the size of the image tensor generated from the decoder path.

    Args:
        input_tensor ([tensor]): [The image tensor whose size is to be cropped.]
        target_tensor ([tensor]): [The image tensor whose size is to be cropped to.]

    Returns:
        [input_tensor]: [The input tensor cropped to the target size.]
    """
    target_size = target_tensor.size()[2]
    tensor_size = input_tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return input_tensor[:, :, delta: tensor_size - delta, delta: tensor_size - delta]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Define the MaxPooling 2x2 layer with stride 2.
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the 5 downwards convolutions. Number of input and output channels are specified on the paper.
        self.convolution_1 = double_convolution(1, 64)
        self.convolution_2 = double_convolution(64, 128)
        self.convolution_3 = double_convolution(128, 256)
        self.convolution_4 = double_convolution(256, 512)
        self.convolution_5 = double_convolution(512, 1024)

        # Define the 5 upwards transpostions and forward convolutions.
        sizes = [1024, 512, 256, 128]

        self.up_transpose_1 = up_transpose(1024, 512)
        self.up_conv_1 = double_convolution(1024, 512)

        self.up_transpose_2 = up_transpose(512, 256)
        self.up_conv_2 = double_convolution(512, 256)

        self.up_transpose_3 = up_transpose(256, 128)
        self.up_conv_3 = double_convolution(256, 128)

        self.up_transpose_4 = up_transpose(128, 64)
        self.up_conv_4 = double_convolution(128, 64)

        # Output layer.
        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        # out_ch=2 because there is only 2 possible classes i.e. background or mole


    def forward(self, image):
        """Forward passing function.
        ENCODER PART:
        Downsample the input image by passing it through all the convolutions and maxpools.
        DECODER PART:
        Up-transpose the image by upscaling and passing through the dodouble convolutions.

        Args:
            image ([tensor]): [Tensor containing information about (channels, batches, height,width) of an image.]

        Returns:
            [final]: [The final tensor after passing through the network.]
        """
        # ENCODER
        # 1st convolutional block
        down_conv_1 = self.convolution_1(image)
        maxpool_1 = self.maxpool(down_conv_1)
        # 2nd
        down_conv_2 = self.convolution_2(maxpool_1)
        maxpool_2 = self.maxpool(down_conv_2)
        # 3rd
        down_conv_3 = self.convolution_3(maxpool_2)
        maxpool_3 = self.maxpool(down_conv_3)
        # 4th
        down_conv_4 = self.convolution_4(maxpool_3)
        maxpool_4 = self.maxpool(down_conv_4)
        # 5th. There is no maxpooling after the last double convolution.
        down_conv_5 = self.convolution_5(maxpool_4)
        print(down_conv_5.size())

        # DECODER
        # 1st up-convolutional block
        up_trans_1 = self.up_transpose_1(down_conv_5)
        crop_1 = crop_img(down_conv_4, up_trans_1)
        up_convo_1 = self.up_conv_1(torch.cat([up_trans_1, crop_1], 1))

        # 2nd up-convolutional block
        up_trans_2 = self.up_transpose_2(up_convo_1)
        crop_2 = crop_img(down_conv_3, up_trans_2)
        up_convo_2 = self.up_conv_2(torch.cat([up_trans_2, crop_2], 1))

        # 3rd up-convolutional block
        up_trans_3 = self.up_transpose_3(up_convo_2)
        crop_3 = crop_img(down_conv_2, up_trans_3)
        up_convo_3 = self.up_conv_3(torch.cat([up_trans_3, crop_3], 1))

        # 4th up-convolutional block
        up_trans_4 = self.up_transpose_4(up_convo_3)
        crop_4 = crop_img(down_conv_1, up_trans_4)
        up_convo_4 = self.up_conv_4(torch.cat([up_trans_4, crop_4], 1))

        # final convolutional block
        final = self.out(up_convo_4)

        print(final.size())
        return final


if __name__ == "__main__":
    image = torch.rand((1, 1, 572, 572))
    model = UNet()
    print(model(image))
