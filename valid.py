import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
import os
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Resnetblock(nn.Module):
    """
    resnet block
    """
    def __init__(self,channels):
        super(Resnetblock,self).__init__()#for the current problem ,channel =1
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=(3,3),stride=1,padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=(3,3),stride=1,padding=1)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self,x):
        residual = x
        x1 = self.conv1(x)
        x2 = self.norm1(x1)
        x3 = self.relu(x2)
        x3 = self.conv2(x3)
        x4 = self.norm2(x3)
        x4 += residual
        out = self.relu(x4)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2d(3,3,kernel_size=(7,7),stride=1,padding=3)
        self.norm1 = nn.InstanceNorm2d(3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(3,3,kernel_size=(7,7),stride=1,padding=3)
        self.norm2 = nn.InstanceNorm2d(3)
        self.conv3 = nn.Conv2d(3,3,kernel_size=(3,3),stride=2,padding=1)
        self.norm3 = nn.InstanceNorm2d(3)
        self.conv4 = nn.Conv2d(3,3,kernel_size=(3,3),stride=2,padding=1)
        self.norm4 = nn.InstanceNorm2d(3)
        self.resnet_blocks = nn.Sequential(
            Resnetblock(3),
            Resnetblock(3),
            Resnetblock(3),
            Resnetblock(3),
            Resnetblock(3),
            Resnetblock(3)
        )
    def forward(self,x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.tanh(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.norm4(out)
        out = self.relu(out)
        out = self.resnet_blocks(out)
        return out

class Multipath_alleviation(nn.Module):
    def __init__(self):
        super(Multipath_alleviation,self).__init__()
        self.resnetblocks = nn.Sequential(
            Resnetblock(3),
            Resnetblock(3),
            Resnetblock(3),
            Resnetblock(3),
            Resnetblock(3),
            Resnetblock(3)
        )
        self.conv1 = nn.ConvTranspose2d(3,3,kernel_size=(3,3),stride=2,padding=1)
        self.norm1 = nn.InstanceNorm2d(3)
        self.conv2 = nn.ConvTranspose2d(3, 3, kernel_size=(3, 3), stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(3,3,kernel_size=(7,7),stride=1,padding =3)
        self.norm3 = nn.InstanceNorm2d(3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.resnetblocks(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.sigmoid(out)
        return out

class Localization(nn.Module):
    def __init__(self):
        super(Localization,self).__init__()
        self.resnetblocks = nn.Sequential(
            Resnetblock(3),
            Resnetblock(3),
            Resnetblock(3)
        )
        self.conv1 = nn.ConvTranspose2d(3,3,kernel_size=(3,3),stride=2,padding=1)
        self.norm1 = nn.InstanceNorm2d(3)
        self.conv2 = nn.ConvTranspose2d(3, 3, kernel_size=(3, 3), stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(3,3,kernel_size=(7,7),stride=1,padding =3)
        self.norm3 = nn.InstanceNorm2d(3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.resnetblocks(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.sigmoid(out)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()

encoder = Encoder()
encoder.load_state_dict(torch.load(f'./weights_per_epoch/encoder_weights_50.pth'))
#encoder =encoder.to(device)
localization_decoder = Localization()
localization_decoder.load_state_dict(torch.load(f'./weights_per_epoch/localization_weights_50.pth'))

def valid(index):
    error_list = []
    try:
        doa_extract = torch.load(f'./doa_train/doa_extract_{index}.pt')
    except:
        return 'The file is broken.'
    target2 = torch.load(f'./target2/target2_{index}.pt')
    doa_extract = torch.transpose(doa_extract,0,1)
    #doa_extract = doa_extract.to(device)
    #target2 = target2.to(device)
    for input_frame,outpt_frame in zip(doa_extract,target2):
        outpt_frame = outpt_frame.repeat(3, 1, 1)
        input_frame = torch.unsqueeze(input_frame,0)
        outpt_frame = torch.unsqueeze(outpt_frame, 0)
        input_frame = F.pad(input_frame, (0, 3, 0, 1, 0, 0, 0, 0))
        outpt_frame = F.pad(outpt_frame, (0, 3, 0, 1, 0, 0, 0, 0))
        input_frame = torch.tensor(input_frame,dtype=torch.float)
        outpt_frame = torch.tensor(outpt_frame,dtype=torch.float)
        encoded = encoder(input_frame)
        location_hat = localization_decoder(encoded)
        error = criterion(location_hat,outpt_frame)
        error = error.to('cpu').detach().numpy().item()
        error_list.append(error)
        location_hat.to('cpu').detach().numpy()
        outpt_frame.to('cpu').detach().numpy()
    return error_list,location_hat,outpt_frame


a,b,c = valid(20)
b = b.detach().numpy()
c = c.detach().numpy()