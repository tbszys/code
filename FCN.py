from PIL import Image
import torch.nn as nn
import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from torchvision.models import vgg16
import joblib

X=[]
y=[]
for i in range(1,329):
    if i<10:
        img=Image.open('D:\\FCN\\FCN\\weizmann_horse_db\\rgb\\horse'+str(i)+'.jpg')
        img2=Image.open('D:\\FCN\\FCN\\weizmann_horse_db\\figure_ground\\horse'+str(i)+'.jpg')
    if i>9 and i<100:
        img=Image.open('D:\\FCN\\FCN\\weizmann_horse_db\\rgb\\horse0'+str(i)+'.jpg')
        img2=Image.open('D:\\FCN\\FCN\\weizmann_horse_db\\figure_ground\\horse0'+str(i)+'.jpg')
    else:
        img=Image.open('D:\\FCN\\FCN\\weizmann_horse_db\\rgb\\horse'+str(i)+'.jpg')
        img2=Image.open('D:\\FCN\\FCN\\weizmann_horse_db\\figure_ground\\horse'+str(i)+'.jpg')
    img=np.array(img)
    img2=np.array(img2)
    X.append(img)
    y.append(img2)
    




x_train=X[0:279]
y_train=y[0:279]
x_val=X[279:328]
y_val=y[279:328]
#print(y[0])

class VGG(nn.Module):
    def __init__(self, pretrained=False):
        super(VGG, self).__init__()

        # conv1 1/2
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2 1/4
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv3 1/8
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        #x = self.relu1_2(self.conv1_2(x))
        x = self.pool1(x)
        pool1 = x

        x = self.relu2_1(self.conv2_1(x))
        #x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)
        pool2 = x

        x = self.relu3_1(self.conv3_1(x))
        #x = self.relu3_2(self.conv3_2(x))
        #x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)
        pool3 = x


        return pool1, pool2,pool3

class FCN8s(nn.Module):
    def __init__(self, num_classes, backbone="vgg"):
        super(FCN8s, self).__init__()
        self.num_classes = num_classes
        if backbone == "vgg":
            self.features = VGG()



        # deconv1 1/4
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # deconv1 1/2
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        # deconv1 1/1
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.features(x)


        y = self.bn3(self.relu3(self.deconv3(features[2]))+features[1])

        y = self.bn4(self.relu4(self.deconv4(y)))

        y = self.bn5(self.relu5(self.deconv5(y)))

        y = self.classifier(y)

        
        return torch.nn.functional.sigmoid(y)

def fast_hist(a, b, n):
       
         k = (a >= 0) & (a < n) 
         
         return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n) 

         
def meanIntersectionOverUnion(confusionMatrix):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
         intersection = np.diag(confusionMatrix) # 取对角元素的值，返回列表
         union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表 
         IoU = intersection / union  # 返回列表，其值为各个类别的IoU
         mIoU = np.nanmean(IoU) # 求各类别IoU的平均
         return mIoU

if __name__ == "__main__":
    batch_size, num_classes, h, w = 1, 1, 64, 64

   

    model = FCN8s(num_classes)
    #y = model(x)
    #loss = torch.nn.CrossEntropyLoss()
    
    loss = torch.nn.BCELoss()
    #learnstep = 0.0000001
    #optim = torch.optim.SGD(model.parameters(),lr=learnstep)
    optim = torch.optim.Adam(model.parameters(),
                lr=0.0001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)

    epoch = 10
    pic_loss=[]
    pic_loss_val=[]
    pic_miou_val=[]

    train_step = 0 #每轮训练的次数
    model.train()#模型在训练状态
    for i in range(epoch):
        print("第{}轮训练".format(i+1))
        train_step = 0
        Loss=[]
        for data in range(0,279):
            imgs = x_train[data]
            imgs = torch.from_numpy(imgs).float()
            imgs=imgs.permute(2,0,1)
            imgs = imgs.unsqueeze(0)
            #print(imgs.shape)
            imgs = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
            outputs = model(imgs)
            #outputs = torch.nn.Sigmoid(outputs)
            assert outputs.size() == torch.Size([batch_size, num_classes, h, w])
            #print(outputs[0][0][0][0])
            #print(outputs[0][1][0][0])
            size=outputs.shape
            n=y[data]
            im = Image.fromarray(n)
            im = im.resize((size[3],size[2]),Image.ANTIALIAS)
            
            im=np.array(im)
            im = torch.from_numpy(im).float()
           
            real_out = np.zeros((size[2],size[3]),dtype=float)
            for p in range(0,size[2]):
                for q in range(0,size[3]):
                    
                    real_out[p][q]=outputs[0][0][p][q]
            real_out= torch.from_numpy(real_out).float()
            
            result_loss = loss(real_out,im)
            a_loss=abs(result_loss)
            Loss.append(a_loss)
            optim.zero_grad()
            result_loss.requires_grad_(True)
            result_loss.backward()
            optim.step()

            train_step+=1
            if(train_step%100==0):
                
                print("第{}轮的第{}次训练的loss:{}".format((i+1),train_step,result_loss.item()))
        Loss=torch.Tensor(Loss)
        Loss=Loss.detach().numpy()
        print("一个epoch的损失",np.mean(Loss))
        pic_loss.append(np.mean(Loss))
        model.eval() #在验证状态
        test_total_loss = 0
        Loss_val=[]
        Miou=[]
        with torch.no_grad(): # 验证的部分
            for test_data  in range(0,49):
                imgs2 = x_val[test_data]
                imgs2 = torch.from_numpy(imgs2).float()
                imgs2=imgs2.permute(2,0,1)
                imgs2 = imgs2.unsqueeze(0)
            #print(imgs.shape)
                imgs2 = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
                outputs2 = model(imgs2)
                size2=outputs2.shape
                n_val=y_val[test_data]
                im_val = Image.fromarray(n_val)
                im_val = im_val.resize((size2[3],size2[2]),Image.ANTIALIAS)
           
                im_val=np.array(im_val)
                im_val = torch.from_numpy(im_val).float()
                
                real_out_val = np.zeros((size2[2],size2[3]),dtype=float)
                out = np.zeros((size2[2],size2[3]),dtype=float)
                for p in range(0,size2[2]):
                   for q in range(0,size2[3]):
                    
                    real_out_val[p][q]=outputs2[0][0][p][q]
                    if real_out_val[p][q]>0.5:
                        out[p][q]=1
                    else:
                        out[p][q]=0
                real_out_val= torch.from_numpy(real_out_val).float()
                out = torch.from_numpy(out).float()
                
                test_result_loss=loss(real_out_val,im_val)
                im_val1 = np.zeros((size2[2],size2[3]),dtype=int)
                out1 = np.zeros((size2[2],size2[3]),dtype=int)
                for p in range(0,size2[2]):
                    for q in range(0,size2[3]):
                        if out[p][q]>0.5:
                            out1[p][q]=1
                        else:
                            out1[p][q]=0
                        if im_val[p][q]>0.5:
                            im_val1[p][q]=1
                        else:
                            im_val1[p][q]=0
                confusionMatrix=fast_hist(out1.reshape(1,-1),im_val1.reshape(1,-1),2)
                miou=meanIntersectionOverUnion(confusionMatrix)
                aval_loss=abs(test_result_loss)
                Loss_val.append(aval_loss)
                Miou.append(miou)
        Loss_val=torch.Tensor(Loss_val)
        Loss_val=Loss_val.detach().numpy()
        
        print("在验证集上的损失为",np.mean(Loss_val))
        pic_loss_val.append(np.mean(Loss_val))
        Miou=torch.Tensor(Miou)
        Miou=Miou.detach().numpy()
        
    
        print("在验证集上的miou为",np.mean(Miou))
        pic_miou_val.append(np.mean(Miou))
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
plt.figure()
plt.plot(pic_loss,'b',label = '训练集损失')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.figure()
plt.plot(pic_loss_val,'b',label = '验证集损失')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.figure()
plt.plot(pic_miou_val,'b',label = 'miou')
plt.ylabel('miou')
plt.xlabel('epoch')
plt.legend()
plt.show()

joblib.dump(model,'D:\FCN\FCN\clf.pkl') #将clf存入.pkl的文件中