import os, sys
import traceback

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import time
from myimgfolder import TrainImageFolder
from colornet import ColorNet #,VideoNetb
#from Dataset import MomentsInTimeDataset
import argparse
import torch.nn as nn

### custom lib
sys.path.append('./networks')
from resample2d_package.modules.resample2d import Resample2d
import networks

original_transform = transforms.Compose([
    transforms.Scale(200),
    transforms.RandomCrop(192),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

have_cuda = torch.cuda.is_available()
epochs = 1
startTime = time.time()
print(startTime)
data_dir = "/Volume2/tpanambur/Deep Clustering/MITMB/data/Moments_in_Time_Mini/trainingImgs/"

train_set = TrainImageFolder(data_dir, original_transform)
train_set_size = len(train_set)
train_set_classes = train_set.classes
train_loader = torch.utils.data.DataLoader(train_set, batch_size=75, shuffle=True, num_workers=0)
print('The length of train loader is', len(train_loader))
print('The length of train loader dataset is',len(train_loader.dataset))
color_model = ColorNet()
#video_model = VideoNet()
if os.path.exists('colornet_params.pkl'):
    print('Loaded colornet')
    color_model.load_state_dict(torch.load('colornet_params.pkl'))
if have_cuda:
    color_model.cuda()
    #video_model.cuda()
optimizer = optim.Adadelta(color_model.parameters())

print('loadtime',time.time()-startTime)

def train(epoch):
    color_model.train()
    #video_model.train()
    ### Load pretrained FlowNet2
    parser = argparse.ArgumentParser()
    opts = parser.parse_args()
    #opts = {}
    opts.rgb_max = 1.0
    opts.fp16 = False

    FlowNet = networks.FlowNet2(opts, requires_grad=False)
    model_filename = os.path.join("pretrained_models", "FlowNet2_checkpoint.pth.tar")
    print("===> Load %s" %model_filename)
    checkpoint = torch.load(model_filename)
    FlowNet.load_state_dict(checkpoint['state_dict'])
    if have_cuda:
        FlowNet = FlowNet.cuda()
        flow_warping = Resample2d().cuda()
     ### criterion and loss recorder
    opts.loss = 'L1'
    if opts.loss == 'L2':
        criterion = nn.MSELoss(size_average=True)
    elif opts.loss == 'L1':
        criterion = nn.L1Loss(size_average=True)
    else:
        raise Exception("Unsupported criterion %s" %opts.loss)
        
    

    try:
        for batch_idx, (data, classes) in enumerate(train_loader):
            torch.cuda.empty_cache()
            ST_loss = 0
            LT_loss = 0
            messagefile = open('./message.txt', 'a')
            original_img = data[0].unsqueeze(1).float()
            img_ab = data[1].float()
            img_color = data[2].float()
	
            if have_cuda:
                original_img = original_img.cuda()
                img_ab = img_ab.cuda()
                classes = classes.cuda()
                img_color = img_color.cuda()

            original_img = Variable(original_img)
            img_ab = Variable(img_ab)
            classes = Variable(classes)
            #img_color = Variable(img_color)
            optimizer.zero_grad()
            class_output, output = color_model(original_img, original_img)
            
        
            '''output_t = torch.zeros(original_img.shape[0],3,original_img.shape[2],original_img.shape[3])
            if have_cuda:
                output_t = output_t.cuda()
            output_t[0:1] = torch.cat((original_img[0:1],original_img[0:1],original_img[0:1]),1)
            output_t[1:] = torch.cat((original_img[1:],output[:-1]),1)
            class_output, output_v = video_model(output_t, output_t)
            output_color = Variable(torch.cat((original_img, output_v), 1))
            #ems_loss_v = torch.pow((img_ab - output_v), 2).sum() / torch.from_numpy(np.array(list(output.size()))).prod().float().cuda()
            
            #output_color = torch.from_numpy(output_color.data.cpu().numpy().transpose((0, 2, 3, 1))).cuda()'''
            output_color = Variable(torch.cat((original_img, output), 1))
            opts.alpha = 50.0
            opts.w_ST = 100
            
            for i in range(1,len(original_img)):
                frame_i1 = img_color[i-1:i]
                frame_i2 = img_color[i:i+1]
                frame_o1 = output_color[i-1:i].detach()
                frame_o1.requires_grad = False
                frame_o2 = output_color[i:i+1]
                ### short-term temporal loss
                if opts.w_ST > 0:
                    ### compute flow (from I2 to I1)
                    flow_i21 = FlowNet(frame_i2, frame_i1)      
                    ### warp I1 and O1
                    warp_i1 = flow_warping(frame_i1, flow_i21)
                    warp_o1 = flow_warping(frame_o1, flow_i21)
                    ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 ) 
                    noc_mask2 = torch.exp( -opts.alpha * torch.sum(frame_i2 - warp_i1, dim=1).pow(2) ).unsqueeze(1)
                    ST_loss += opts.w_ST * criterion(frame_o2 * noc_mask2, warp_o1 * noc_mask2)
              
            ### long-term temporal loss
            opts.w_LT = 100
            
            if opts.w_LT > 0:

                t1 = 0
                for t2 in range(t1 + 2, len(original_img)):

                    frame_i1 = img_color[t1:t1+1]
                    frame_i2 = img_color[t2:t2+1]

                    frame_o1 = output_color[t1:t1+1].detach() ## make a new Variable to avoid backwarding gradient
                    frame_o1.requires_grad = False

                    frame_o2 = output_color[t2]

                    ### compute flow (from I2 to I1)
                    flow_i21 = FlowNet(frame_i2, frame_i1)
                    
                    ### warp I1 and O1
                    warp_i1 = flow_warping(frame_i1, flow_i21)
                    warp_o1 = flow_warping(frame_o1, flow_i21)

                    ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
                    noc_mask2 = torch.exp( -opts.alpha * torch.sum(frame_i2 - warp_i1, dim=1).pow(2) ).unsqueeze(1)

                    LT_loss += opts.w_LT * criterion(frame_o2 * noc_mask2, warp_o1 * noc_mask2)

                ### end of t2
            ### end of w_LT
            
            #ems_loss_v = torch.pow((img_ab - output_v), 2).sum() / torch.from_numpy(np.array(list(output.size()))).prod().float().cuda()
            ems_loss_i = torch.pow((img_ab - output), 2).sum() / torch.from_numpy(np.array(list(output.size()))).prod().float().cuda()

            cross_entropy_loss = 1/300 * F.cross_entropy(class_output, classes)
            
            

            loss = ems_loss_i  + 0.000001 * ST_loss + 0.000001 * LT_loss
            lossmsg = 'Ems_loss_v: %.9f\t' % (ems_loss_i)+'ST_loss: %.9f\t' % (ST_loss)+'LT_loss: %.9f\t' % (LT_loss)+'loss: %.9f\n' % (loss.data)
            messagefile.write(lossmsg)
            #videoLoss = ems_loss_v + ST_loss + LT_loss
            loss.backward(retain_graph=True)
            #ems_loss_i.backward()
            
            cross_entropy_loss.backward()
            optimizer.step()
            if batch_idx % 1000 == 0:
                message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data)
                messagefile.write(message)
                torch.save(color_model.state_dict(), 'colornet_params.pkl')
                #torch.save(video_model.state_dict(), 'videonet_params.pkl')
            messagefile.close()
            if batch_idx % 5000 == 0:
                print('Elapsed: ',time.time()-startTime)
                # print('Train Epoch: {}[{}/{}({:.0f}%)]\tLoss: {:.9f}\n'.format(
                #     epoch, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss.data[0]))
    except Exception:
        logfile = open('log.txt', 'w')
        logfile.write(traceback.format_exc())
        logfile.close()
    finally:
        torch.save(color_model.state_dict(), 'colornet_params.pkl')
        #torch.save(video_model.state_dict(), 'videonet_params.pkl')


for epoch in range(1, epochs + 1):
    train(epoch)

print(time.time()-startTime)
