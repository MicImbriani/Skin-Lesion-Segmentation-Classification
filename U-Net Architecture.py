import torch 
import torch.nn as nn

# This architecture is based exatly on the description provided in 
# the section 2 "Network Architecture" of the U-Net paper.


# The architecture consists of the repeated application of two 3x3 unpadded covolutions.
# Thus I will abstract and create a function that can be applied to all repetitions.
# It takes 2 parameters: input channels and output channels. 
def double_convolution(in_ch, out_ch):
    # It is a normal convolutional network, hence sequential structure.
    convolution = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3)
        # inplace=True because there is no need to create and store a copy
        nn.ReLU(inplace=True)
        nn.Conv2d(out_ch, out_ch, kernel_size=3)
        nn.ReLU(inplace=True)
    )
    return convolution

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Define the MaxPooling 2x2 layer with stride 2.
        self.maxpool =  nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the 5 convolutions. Number of input and output channels are specified on the paper.
        self.convolution_1 = double_convolution(1, 64)
        self.convolution_2 = double_convolution(64, 128)
        self.convolution_3 = double_convolution(128, 256)
        self.convolution_4 = double_convolution(256, 512)
        self.convolution_5 = double_convolution(512, 1024)

    # Define a function that will downsample the input image by passing it through all the convolutions and maxpools.
    # Here a "floor" is every 2 conv layers and 1 maxpool layer.
    def forward(self, image):
        # 1st floor of the U-Net Architecture
        first_conv = self.convolution_1(image)
        first_maxpool = self.maxpool(first_conv)
        # 2nd
        second_conv = self.convolution_2(first_maxpool)
        second_maxpool = self.maxpool(second_conv)
        # 3rd
        third_conv = self.convolution_3(second_maxpool)
        third_maxpool = self.maxpool(third_conv)
        # 4th
        fourth_conv = self.convolution_4(third_maxpool)
        fourth_maxpool = self.maxpool(fourth_conv)
        # 5th. There is no maxpooling after the last double convolution.
        fifth_conv = self.convolution_5(fourth_maxpool)
        print(fifth_conv.size())
        


        if __name__ == "__main__":
            image = torch.rand((1,1,572,572))
            model = UNet()
            print(model(image))