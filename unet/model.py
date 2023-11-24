import torch
from torch import nn
from torch.nn import (Conv2d, ConvTranspose2d, MaxPool2d, Module, ModuleList,
                      ReLU)
from torch.nn import functional as F
from torchvision.transforms import CenterCrop

import config


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # store the convolution and RELU layers
        self.layers = ModuleList(
            [
                Conv2d(in_channels, out_channels, kernel_size=3),
                ReLU(inplace=True),
                Conv2d(out_channels, out_channels, kernel_size=3),
                ReLU(inplace=True),
            ]
        )

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()

        # store the encoder blocks and maxpooling layer
        self.enc_blocks = ModuleList(
            [
                Block(in_channels=channels[i], out_channels=channels[i + 1])
                for i in range(len(channels) - 1)
            ]
        )

        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        block_outputs = []

        # loop through the encoder blocks
        for block in self.enc_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            print(x.shape)
            block_outputs.append(x)
            x = self.pool(x)

        # return the list containing the intermediate outputs
        return block_outputs


class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels

        self.upconvs = ModuleList(
            [
                ConvTranspose2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=2,
                    stride=2,
                )
                for i in range(len(channels) - 1)
            ]
        )

        self.dec_blocks = ModuleList(
            [
                Block(in_channels=channels[i], out_channels=channels[i + 1])
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x, enc_features):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)

            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block

            temp = self.crop(enc_features[i], x)
            print(enc_features[i].shape, x.shape)
            x = torch.cat([temp, x], dim=1)
            x = self.dec_blocks[i](x)

        # return the final decoder output
        return x

    def crop(self, enc_features, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        enc_features = CenterCrop([H, W])(enc_features)

        # return the cropped features
        return enc_features


class UNet(Module):
    def __init__(
        self,
        enc_channels=(3, 16, 32, 64),
        dec_channels=(64, 32, 16),
        n_classes=1,
        retain_dim=True,
        out_size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
    ):
        super().__init__()

        # initialize the encoder and decoder
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

        # initialize the regression head and store the class variables
        self.head = Conv2d(dec_channels[-1], n_classes, kernel_size=1)
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x):
        # grab the features from the encoder
        enc_features = self.encoder(x)

        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        dec_features = self.decoder(enc_features[-1], enc_features[::-1][1:])

        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(dec_features)

        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retain_dim:
            map = F.interpolate(map, self.out_size)

        # return the segmentation map
        return map
