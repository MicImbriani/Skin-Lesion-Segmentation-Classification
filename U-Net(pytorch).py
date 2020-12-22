import torch 
import torch.nn as nn

# This architecture is based exatly on the description provided in 
# the section 2 "Network Architecture" of the U-Net paper.


 
def double_convolution(in_ch, out_ch):
    """The architecture consists of the repeated application of two 3x3 unpadded covolutions.
    Each convolution is followed by a ReLU. 

    Args:
        in_ch ([int]): [Number of input channels]
        out_ch ([int]): [Number of output channels]

    Returns:
        [nn.Sequential]: [The callableobject for applying the convolution]
    """    
    # It is a normal convolutional network, hence sequential structure.
    convolution = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3),
        # inplace=True because there is no need to create and store a copy
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return convolution



def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta ]



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Define the MaxPooling 2x2 layer with stride 2.
        self.maxpool =  nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the 5 downwards convolutions. Number of input and output channels are specified on the paper.
        self.convolution_1 = double_convolution(1, 64)
        self.convolution_2 = double_convolution(64, 128)
        self.convolution_3 = double_convolution(128, 256)
        self.convolution_4 = double_convolution(256, 512)
        self.convolution_5 = double_convolution(512, 1024)

        # Define the 5 upwards transpostions with for loop.
        sizes = [1024, 512, 256, 128]

        self.up_transpose_1 = nn.ConvTranspose2d(
        in_channels=1024,
        out_channels=512,
        kernel_size=2,
        stride=2)
        self.up_conv_1 = double_convolution(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(
        in_channels=512,
        out_channels=256,
        kernel_size=2,
        stride=2)
        self.up_conv_2 = double_convolution(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(
        in_channels=256,
        out_channels=128,
        kernel_size=2,
        stride=2)
        self.up_conv_3 = double_convolution(256, 128)

        self.up_transpose_4 = nn.ConvTranspose2d(
        in_channels=128,
        out_channels=64,
        kernel_size=2,
        stride=2)
        self.up_conv_4 = double_convolution(128, 64)
        
        # Output layer.
        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)     
        #out_ch=2 because there is only 1 element to be segmented




    def forward(self, image):
        """Standard forward passing function.
        ENCODER PART:
        I downsample the input image by passing it through all the convolutions and maxpools. 
        Here a "floor" is every 2 conv layers and 1 maxpool layer.
        DECODER PART:
        Up-transpose the image

        Args:
            image ([tensor]): [Tensor containing information about (channels, batches, height,width) of an image]
        """        
        # ENCODER
        # 1st floor of the U-Net Architecture
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
        up_trans_1 = self.up_transpose_1(down_conv_5)
        crop_1 = crop_img(down_conv_4, up_trans_1)
        up_convo_1 = self.up_conv_1(torch.cat([up_trans_1, crop_1], 1))
        
        up_trans_2 = self.up_transpose_2(up_convo_1)
        crop_2 = crop_img(down_conv_3, up_trans_2)
        up_convo_2 = self.up_conv_2(torch.cat([up_trans_2, crop_2], 1))
        print(up_convo_2.size())

        up_trans_3 = self.up_transpose_3(up_convo_2)
        crop_3 = crop_img(down_conv_2, up_trans_3)
        up_convo_3 = self.up_conv_3(torch.cat([up_trans_3, crop_3], 1))

        up_trans_4 = self.up_transpose_4(up_convo_3)
        crop_4 = crop_img(down_conv_1, up_trans_4)
        up_convo_4 = self.up_conv_4(torch.cat([up_trans_4, crop_4], 1))

        final = self.out(up_convo_4)
        print(final.size())
        return final


if __name__ == "__main__":
    image = torch.rand((1,1,572,572))
    model = UNet()
    print(model(image))