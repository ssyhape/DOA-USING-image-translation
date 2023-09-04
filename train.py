import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
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
    def __init__(self,batchsize):
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
    def __init__(self,batchsize):
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
    def __init__(self,batchsize):
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




batchsize = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()

encoder = Encoder(batchsize)
encoder = encoder.to(device)
multipath_decoder = Multipath_alleviation(batchsize)
multipath_decoder = multipath_decoder.to(device)
localization_decoder = Localization(batchsize)
localization_decoder = localization_decoder.to(device)
optimizer = optim.Adam(list(encoder.parameters())+list(multipath_decoder.parameters())+list(localization_decoder.parameters()),lr = 0.00001,weight_decay=0.00001)

def train(doa_extracted_data,compensation_data,localization_data):
    optimizer.zero_grad()
    encoded = encoder(doa_extracted_data)
    multipath_output = multipath_decoder(encoded)
    localization_output = localization_decoder(encoded)
    L1_lambda = 0.0005
    l1_regularization = torch.tensor(0.)
    l1_regularization = l1_regularization.to(device)
    for param in localization_decoder.parameters():
        l1_regularization += torch.norm(param, p=1)

    loss = torch.tensor(0.)
    loss = loss.to(device,dtype=torch.float)
    for channel in range(3):
        loss += (1/3)*criterion(multipath_output[:,channel,:,:],compensation_data[:,channel,:,:]) + criterion(localization_output[:,channel,:,:],localization_data[:,channel,:,:])
    loss += L1_lambda*l1_regularization

    loss.backward()
    optimizer.step()
    return loss.item()

def get_data(index,input_fold,target1_fold,target2_fold):
    file1 = f'doa_extract_{index}.pt'
    try:
        X = torch.load(input_fold + file1)
    except:
        return 'error'
    file2 = f'doa_extract_compensation_{index}.pt'
    X  = torch.transpose(X, 0, 1)
    try:
        Y_1 = torch.load(target1_fold + file2)
    except:
        return 'error'
    Y_1 = torch.transpose(Y_1, 0, 1)
    file3 = f'target2_{index}.pt'
    try:
        Y_2 = torch.load(target2_fold + file3)
    except:
        return 'error'
    Y_2 = Y_2.unsqueeze(0).repeat(3, 1, 1, 1)
    Y_2 = torch.transpose(Y_2,0,1)
    X = X.to(device,dtype=torch.float)
    Y_1 = Y_1.to(device,dtype=torch.float)
    Y_2 = Y_2.to(device,dtype = torch.float)

    train_ds = TensorDataset(X, Y_1,Y_2)
    train_dl = DataLoader(train_ds, batch_size=batchsize)

    return train_dl

if __name__ == "__main__":
    num_epochs = 150
    doa_filefold_name = f'./doa_train/'
    target1_filefold_name = f'./target1/'
    target2_filefold_name = f'./target2/'
#    torch.save(encoder.state_dict(), f'./weights_per_epoch/encoder_weights_1.pth')
    loss_per_epoch =[]
    for epoch in range(num_epochs):
        total_loss =0
        for ind in range(108):
            train_dataloader = get_data(ind,doa_filefold_name,target1_filefold_name,target2_filefold_name)
            if train_dataloader=='error':
                continue
            for input,target1,target2 in train_dataloader:
                input = F.pad(input,(0,3,0,1,0,0,0,0))
                target1 = F.pad(target1, (0,3,0,1,0,0,0,0))
                target2 = F.pad(target2, (0,3,0,1,0,0,0,0))
                loss = train(input,target1,target2)
                total_loss +=loss
            avg_loss = total_loss / (len(train_dataloader)*(ind+1))
            print(f"epoch {epoch} , index {ind} has finished,error is {avg_loss}")
        loss_per_epoch.append(total_loss/70000)
        torch.save(encoder.state_dict(),f'./weights_per_epoch/encoder_weights_{epoch}.pth')
        torch.save(multipath_decoder.state_dict(),f'./weights_per_epoch/mutipath_weights_{epoch}.pth')
        torch.save(localization_decoder.state_dict(),f'./weights_per_epoch/localization_weights_{epoch}.pth')
        print(f"epoch {epoch} has finished.")
