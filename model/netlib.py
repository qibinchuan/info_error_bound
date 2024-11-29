import torch
import torch.nn as nn
import torch.nn.functional as F


class Basicblock(nn.Module):
    def __init__(self, in_planes, planes, stride, is_res):
        super(Basicblock, self).__init__()
        self.is_res=is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            #nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
                #nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.is_res==True: # 默认是res, 否则是conv级联
            out += self.shortcut(x)
        elif self.is_res==False:
            pass
        else:
            print("Wrong param")
        out = F.relu(out)
        return out

# Resnet
class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes, is_res=True):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block1 = self._make_layer(block, 16, num_block[0], stride=1, is_res=is_res)
        self.block2 = self._make_layer(block, 32, num_block[1], stride=2, is_res=is_res)
        self.block3 = self._make_layer(block, 64, num_block[2], stride=2, is_res=is_res)
        # self.block4 = self._make_layer(block, 512, num_block[3], stride=2)

        self.outlayer = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_block, stride, is_res):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride, is_res))
            else:
                layers.append(block(planes, planes, 1, is_res))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.block1(x)                       # [200, 64, 28, 28]
        x = self.block2(x)                       # [200, 128, 14, 14]
        x = self.block3(x)                       # [200, 256, 7, 7]
        # out = self.block4(out)
        x = F.avg_pool2d(x, 7)                   # [200, 256, 1, 1]
        x = x.view(x.size(0), -1)                # [200,256]
        out = self.outlayer(x)
        return out
    
# 混合模型    
class SNet(nn.Module):
    def __init__(self, n_F, is_res):
        super(SNet, self).__init__()
        # 修改输入通道数为1
        self.n_F = n_F
        self.n_W = 10
        self.is_res = is_res
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.fcs=[]
        self.fc1 = nn.Linear(8*14*14, self.n_W)
        self.fcs.append(self.fc1)
        for F_idx in range(1,self.n_F+1):
            fc = nn.Linear(self.n_W,self.n_W)            
            setattr(self,'fc%i'%F_idx,fc)    
            #将该层添加到这个Module中,setattr函数用来设置属性,
            #其中第一个参数为继承的类别,第二个为名称,第三个是数值
            self.fcs.append(fc)
        self.fcn = nn.Linear(self.n_W,10)
        self.fcs.append(self.fcn)
        
        self.pool = nn.MaxPool2d(2, 2)
        #print(self.fcs)
        self.shortcut = nn.Sequential()



    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = out.view(-1, 8 * 14 * 14)
        #print(self.is_res)
        for F_idx in range(self.n_F+1):
            if F_idx>0 and self.is_res==True:
                #print(self.shortcut(out).shape)
                out = F.relu(self.fcs[F_idx](out))+self.shortcut(out)
            else:
                out = F.relu(self.fcs[F_idx](out))
        out = self.fcs[self.n_F](out)
        #return F.softmax(out, dim=1)
        return out #F.log_softmax(out, dim=1)

# 完全CNN模型
class CNet(nn.Module):
    def __init__(self, n_C):
        super(CNet, self).__init__()
        # 修改输入通道数为1
        self.n_C = n_C
        self.convs=[]
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.convs.append(self.conv1)
        for C_idx in range(2,self.n_C+1):
            conv = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
            setattr(self,'conv%i'%C_idx,conv)    
            #将该层添加到这个Module中,setattr函数用来设置属性,
            #其中第一个参数为继承的类别,第二个为名称,第三个是数值
            self.convs.append(conv)
        self.fc1 = nn.Linear(8*28*28, 10)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #print(self.fcs)

    def forward(self, x):
        for C_idx in range(self.n_C):
            #print(x.shape)
            x = F.relu(self.convs[C_idx](x))
            #print(x.shape)
            x = self.pool(x) # 不加的时候更平滑？
            #print("afetrpool:",x.shape)
        x = x.view(-1, 8 * 28 * 28)
        x = self.fc1(x)
        return x #F.softmax(x, dim=1)

# 完全线性模型
class LNet(nn.Module):
    def __init__(self,n_F,n_W=64):
        super(LNet, self).__init__()
        #xw+b
        self.n_W = n_W
        self.n_F = n_F

        self.fcs=[]
        self.fc1 = nn.Linear(28*28,self.n_W)
        self.fcs.append(self.fc1)
        #self.fc1 = nn.Linear(28*28,256)
        #self.fcs.append(self.fc1)

        #self.fc2 = nn.Linear(256,64)
        #self.fcs.append(self.fc2)
        
        for F_idx in range(1,self.n_F+1):
            fc = nn.Linear(self.n_W,self.n_W)            
            setattr(self,'fc%i'%F_idx,fc)    
            #将该层添加到这个Module中,setattr函数用来设置属性,
            #其中第一个参数为继承的类别,第二个为名称,第三个是数值
            self.fcs.append(fc)
        self.fcn = nn.Linear(self.n_W,10)
        self.fcs.append(self.fcn)
        print("total layer:", len(self.fcs))
 
    def forward(self,x):
        x = x.view(x.size(0),28*28)
        for F_idx in range(len(self.fcs)):
            x = F.relu(self.fcs[F_idx](x))
        return x 



