import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters as p

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x, indices = self.pool(x)
        return x, indices

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, indices, output_size):
        x = self.unpool(x, indices, output_size=output_size)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SegNet(nn.Module):
    def __init__(self, num_classes=p.nb_class):
        super(SegNet, self).__init__()

        # Encoder
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        self.enc_block1 = EncoderBlock(64, 128)
        self.enc_block2 = EncoderBlock(128, 256)
        self.enc_block3 = EncoderBlock(256, 512)
        self.enc_block4 = EncoderBlock(512, 1024)

        # Decoder
        self.dec_block4 = DecoderBlock(1024, 512)
        self.dec_block3 = DecoderBlock(512, 256)
        self.dec_block2 = DecoderBlock(256, 128)
        self.dec_block1 = DecoderBlock(128, 64)
                
        # Classifier
        self.outputs = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        
    def forward(self, x):
        # Encoder
        # print("Encoder")
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x, indices_2 = self.enc_block1(x)
        # print(x.shape)
        x, indices_3 = self.enc_block2(x)
        # print(x.shape)
        x, indices_4 = self.enc_block3(x)
        # print(x.shape)
        x, indices_5 = self.enc_block4(x)
        # print(x.shape)
        
        # Decoder
        # print("Decoder")
        x = self.dec_block4(x, indices_5, output_size=(x.size(2)*2, x.size(3)*2))
        # print(x.shape)
        x = self.dec_block3(x, indices_4, output_size=(x.size(2)*2, x.size(3)*2))
        # print(x.shape)
        x = self.dec_block2(x, indices_3, output_size=(x.size(2)*2, x.size(3)*2))
        # print(x.shape)
        x = self.dec_block1(x, indices_2, output_size=(x.size(2)*2, x.size(3)*2))
        # print(x.shape)
        x = self.outputs(x)
        # print(x.shape)
        outputs = F.softmax(x, dim=1)
        # print(outputs.size())

        return outputs

