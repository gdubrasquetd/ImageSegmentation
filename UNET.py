import torch
import torch.nn as nn
import parameters as p
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.dropout1 = nn.Dropout2d(p.dropout_prob)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.dropout2 = nn.Dropout2d(p.dropout_prob)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        return x
    
class conv_block2(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
       
    
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        
        return x, p
    
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block2(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
    
        
class UNET(nn.Module):
    def __init__(self, num_classes=p.nb_class):
        super().__init__()
        
        #Encoder
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        
        #Bottleneck
        self.b = conv_block(512, 1024)
        
        #Decoder
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        
        #Classifier
        self.outputs = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)

        
    def forward(self, inputs):
        
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        b = self.b(p4)
        
        #Decoder
        # print(b.size())
        d1 = self.d1(b, s4)
        # print(d1.size())
        d2 = self.d2(d1, s3)
        # print(d2.size())
        d3 = self.d3(d2, s2)
        # print(d3.size())
        d4 = self.d4(d3, s1)
        # print(d4.size())
        
        outputs = self.outputs(d4)
        # print(d1.size())
        outputs = F.softmax(outputs, dim=1)

        return outputs