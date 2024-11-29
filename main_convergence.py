import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import netron
import copy


from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


import grad
import os
import numpy as np
import model.netlib as netlib
from config import model_dir
import sys

def tran_onehot(target):
    target_onehot = torch.zeros(target.shape[0],10)
    for idx, label in enumerate(target):
        target_onehot[idx][label]=1
    return target_onehot

def get_log2(value):
    log_v = -50
    if value!=0:
        log_v = np.log2(value)
    return log_v
                    

def save_info(history,model_name):
    fname = model_name+".txt"
    save_dir = "results/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, fname), "w") as f:
        f.write("# {}\n".format(str(fname)))
        for line in history:
            f.write("{:.0f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(*line))    


def val(model, device, val_loader, criterion, epoch, writer):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Compute statistics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #print(predicted)
            #print(targets)
            #sys.exit()

    # Compute average validation loss
    val_loss /= len(val_loader)

    # Log validation loss to TensorBoard
    writer.add_scalar('Val Loss', val_loss, epoch)

    # Compute accuracy
    accuracy = 100. * correct / total

    writer.add_scalar('Accuracy', accuracy, epoch)
    # Print validation statistics
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

def save_model(model, epoch, model_name):
    # Ensure the model save directory exists

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Save the model
    torch.save(model.state_dict(), os.path.join(model_dir,model_name+f'model_epoch_{epoch}.pth'))


def train(model, device, train_loader, optimizer, scheduler, criterion, epochs, writer,test_inputs, test_targets, val_inputs, val_targets):
    model.train()
    total_loss = 0.0
    history = []
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            """
            if batch_idx==0:
                val_inputs[0]=data[0]
                val_targets[0]=target[0]
                #print(val_targets[0])
                #print(target)
            """
            if batch_idx % 2 == 0:
                #print("LR:", optimizer.state_dict()['param_groups'][0]['lr'])
                #cur_val_input = torch.zeros([2]+list(val_inputs.size())[1:])
                #cur_val_target = torch.zeros([2]+list(val_targets.size())[1:],dtype=torch.int64)
                pred_model = copy.deepcopy(model) 
                #pred_model.eval()  # Set the model to evaluation mode
                # 计算基于训练集数据的bound
                optimizer.zero_grad()
                test_outputs = pred_model(test_inputs)
                #pred_model.train()
                cur_loss = criterion(test_outputs, test_targets) # mse 除以了10
                cur_loss.backward()
                gd_norm = grad.cal_grad_norm(pred_model)
                #print(val_outputs)
                _, pred = test_outputs.max(1)
                #print(pred)
                #print(val_targets)
                #print(gd_norm)
                max_e,min_e,F_norm_sq,num_param = grad.compute_jacobian_egienvalue_mul(model, test_inputs)
                #print("max_e{},min_e:{}".format(max_e,min_e))

                #max_e,min_e,jo_beta,num_param = grad.compute_jacobian_egienvalue(model, val_inputs)
                #print("max_e{},min_e:{}".format(max_e,min_e))
                #sys.exit()

                # 计算期望风险
                pred_model = copy.deepcopy(model) 
                optimizer.zero_grad()
                val_outputs = pred_model(val_inputs)
                #pred_model.train()
                cur_loss = criterion(val_outputs, val_targets) # mse 除以了10
                _, pred = val_outputs.max(1)
                val_onehot = torch.zeros([1,10])
                val_onehot[0,val_targets[0]]=1
                fitting_error = torch.zeros([1])
                with torch.no_grad():
                    prob = F.softmax(val_outputs[0],dim=0) # dim = 0,在列上进行Softmax;dim=1,在行上进行Softmax
                    fitting_error = torch.linalg.vector_norm(val_onehot-prob, ord=2)
                    #print(val_onehot[0])
                    #print(prob)
                    #print(fitting_error)
                log_gd, log_min_e, log_max_e, log_cur_loss, log_fitting_error = get_log2(gd_norm.item()**2),get_log2(min_e.item()),get_log2(max_e.item()),get_log2(cur_loss.item()),get_log2(fitting_error)
                history.append([940*epoch+batch_idx, log_gd, log_min_e, log_max_e, log_cur_loss, log_fitting_error])
                if batch_idx % 20 == 0:
                    print("batch_idx:{:.3f}, gd_norm:{:.3f}, min_e:{:.3f},max_e:{:.3f}, U:{:.3f}, L:{:.3f}, fittingerror:{:.3f}".format(batch_idx, log_gd, log_min_e, log_max_e, log_gd-log_min_e, log_gd-log_max_e, 2*log_fitting_error))    

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
        # Log the average training loss for the epoch
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)  # Log the training loss for the epoch
        scheduler.step()
    return history

def main(model_name,is_res,epochs,n_W,total_layer,lr):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    test_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Dataset loading
    #train_dataset = datasets.MNIST(root='./data/tupian', train=True, download=True, transform=transform)
    train_dataset = datasets.MNIST(root='../MNIST', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, 64, shuffle=True)


    #val_dataset = datasets.MNIST(root='./data/test_images', train=False, download=True, transform=test_transform)
    val_dataset = datasets.MNIST(root='../MNIST', train=False, download=True, transform=test_transform)
    val_loader = DataLoader(val_dataset, 1, shuffle=False)

    # prepare the val image
    val_inputs, val_targets,test_inputs, test_targets = None,None,None,None
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if batch_idx==0:
            val_inputs, val_targets = inputs.to(device), targets.to(device)
            print(val_targets)
        else:
            test_inputs, test_targets = inputs.to(device), targets.to(device)
            if val_targets==test_targets:
                print(test_targets)
                print(batch_idx)
                break
    # Initialize the model
    if model_name=="ResNet18":            
        model = netlib.ResNet(netlib.Basicblock, [total_layer-1, 1, 1, 1], 10, is_res).to(device)
    elif model_name=="SNet":
        model = netlib.SNet(total_layer, is_res).to(device)
    elif model_name == "CNet":
        model = netlib.CNet(total_layer).to(device)
    elif model_name == "LNet":
        model = netlib.LNet(total_layer-1,n_W).to(device)
    else:
        print("No such model type!")
        return

    #model.load_state_dict(torch.load("Models/ResNet18model_epoch_1.pth"))

    # Optimizer and loss function    
    print(device)
    print("There are", sum(p.numel() for p in model.parameters()), "parameters.")
    print("There are", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")

    #optimizer = nisgd.NISGD(model.parameters(), lr=0.01, momentum=0.9) 
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-4) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # loss 0.07, acc:18%
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)


    criterion = nn.CrossEntropyLoss() #nn.MSELoss()
    # Initialize TensorBoard writer
    writer = SummaryWriter()
    # Training process
    #val(model, device, val_loader, criterion, epochs, writer)
    #sys.exit()


    summary(model, (1, 28, 28))
    # return 

    history = train(model, device, train_loader, optimizer, scheduler, criterion, epochs, writer,test_inputs, test_targets, val_inputs, val_targets)
    ret_file_name = model_name+"_"+"nores"+"_"+str(epochs)+"_lay"+str(total_layer)
    if is_res==True:
        ret_file_name = model_name+"_"+"res"+"_"+str(epochs)+"_lay"+str(total_layer)
    save_info(history, ret_file_name)
    save_model(model, epochs, model_name)
    val(model, device, val_loader, criterion, epochs, writer)

    # Close the TensorBoard writer
    writer.close()




if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(torch.__version__)
    model_name = "ResNet18"

    # SNet: cnn+linear
    # LNet: linear
    # CNet: cnn
    # ResNet18: resnet
    # 目前使用的是snet 有无res两种结构
    # snet nores 的情况下没有收敛
    is_res=False
    epochs = 1
    lr = 0.002 #renet:0.002 
    n_W = 64
    total_layer = 1
    for model_name in ["LNet"]: #["SNet","ResNet18","LNet","CNet"]:
        for total_layer in range(2,3,1):
            main(model_name,is_res,epochs,n_W,total_layer,lr)
    # acc: 
    # snet_1_nores:63.21%, snet_2_nores:61.09%, snet_3_nores:64.93%, snet_4_nores_4:58.64%
    # snet_5_nores: 60.82%, snet_6_nores: 51.06%, snet_7_nores: 33.93%
    # cnet:97.01%, 
    # resnet18_res_1:96.75%,resnet18_res_2:96.70%, resnet18_res_3: 97.10%, resnet18_res_4: 96.70%
    # resnet188_res_5: 96.55%, resnet18_res_6: 97.13%
    # lnet_1: 66.84%,lnet_1: 52.80%, Lnet_3:
    # snet_1_res:57.35%，snet_2_res:58.43%, snet_3_res:62.68%, snet_4_res:65.28%, snet_5_res:59.68%, snet_6_res: 57.00%, snet_7_res:64.90%, 
