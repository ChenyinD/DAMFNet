import torch
import datasets.dataset as dataset
import model.Net as Net
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
import matplotlib.pyplot as plt



def train():
    # train
    # 超参数
    sig=nn.Sigmoid()
    EPOCH =80  # 前向后向传播迭代次数
    LR = 0.001   # 学习率 learning rate

    f1 = dataset.load_dataset_512(2, '../DAMFNet/datasets/txt/LEVIR-CD/A1.txt', '../DAMFNet/datasets/txt/LEVIR-CD/B1.txt', '../DAMFNet/datasets/txt/LEVIR-CD/C1.txt')
    f2 = dataset.load_dataset_224(2, '../DAMFNet/datasets/txt/LEVIR-CD/A2.txt', '../DAMFNet/datasets/txt/LEVIR-CD/B2.txt', '../DAMFNet/datasets/txt/LEVIR-CD/C2.txt')

    # f1 = dataset.load_dataset_512(2, '../DAMFNet/datasets/txt/WHU-CD/WA.txt', '../DAMFNet/datasets/txt/WHU-CD/WB.txt', '../DAMFNet/datasets/txt/WHU-CD/WC.txt')
    # f2 = dataset.load_dataset_224(2, '../DAMFNet/datasets/txt/WHU-CD/WA2.txt', '../DAMFNet/datasets/txt/WHU-CD/WB2.txt', '../DAMFNet/datasets/txt/WHU-CD/WC2.txt')

    cnn=Net.CDNet().cuda()


    optimizer = torch.optim.Adam(cnn.parameters(),betas=(0.9,0.99), lr=LR,weight_decay=0.0001)  # 定义优化器
    scheduler_1 = LambdaLR(optimizer,lr_lambda=lambda epoch: 0.95**epoch)
    loss_func = nn.BCEWithLogitsLoss().cuda()  # 定义损失函数
    accumulation_steps = 16
    LOSS=[]
    NUM=[]
    crs = []
    confu1 = np.array([[0, 0], [0, 0]])
    i=0
    confu=np.array([[0, 0], [0, 0]])
    for epoch in range(EPOCH):
        for step, (images, images2, label,F1) in enumerate(f1):
            images = images.cuda()
            images = (images - torch.min(images)) / (torch.max(images) - torch.min(images))
            data_copy1 = images.clone().detach()
            images2 = images2.cuda()
            images2 = (images2 - torch.min(images2)) / (torch.max(images2) - torch.min(images2))
            data_copy2 = images2.clone().detach()
            labels = label.cuda()
            seg,d1,d2,d3,d4=cnn(data_copy1,data_copy2)
            CD_loss = loss_func(seg, labels) + loss_func(d1, labels) + loss_func(d2, labels) + loss_func(d3,labels) + loss_func(d4, labels)
            CD_loss.backward()  # 反向传播

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()  # 更新优化器的学习率，一般按照epoch为单位进行更新
                optimizer.zero_grad()  # 清空上一层梯度
            label = labels.detach().cpu().numpy()
            y = sig(seg)
            y = y.detach().cpu().numpy()
            y[y < 0.5] = 0
            converted_image = (y * 255).astype(np.uint8)
            for j in range(2):
             t2, ypredict = cv2.threshold(converted_image[j][0], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
             ytrue = label[j].flatten().astype("int64")
             ypredict = ypredict.flatten()
             ytrue = ytrue.tolist()
             ypredict = ypredict.tolist()
             c = confusion_matrix(ytrue, ypredict)
             if c.shape ==(2,2):
              confu += c
            LOSS.append(CD_loss.data.to('cpu'))
            i+=1
            NUM.append(i)
            print(epoch, 'Epoch: | train loss: %.6f' % CD_loss.data)
        scheduler_1.step()
        torch.save(cnn, 'best_model_new1.pt')
        TP = confu[1][1]
        TN = confu[0][0]
        FN = confu[1][0]
        FP = confu[0][1]
        confu = np.array([[0, 0], [0, 0]])
        if (TP + FN) != 0 and (TP + FP) != 0:
            recall = (TP) / (TP + FN)
            presion = TP / (TP + FP)
            iou = TP / (TP + FN + FP)
            print(presion, recall, iou, '精召差')
        for step, (images, images2,label, F1) in enumerate(f2):
            images = images.cuda()
            images = (images - torch.min(images)) / (torch.max(images) - torch.min(images))
            data_copy1 = images.clone().detach()
            images2 = images2.cuda()
            images2 = (images2 - torch.min(images2)) / (torch.max(images2) - torch.min(images2))
            data_copy2 = images2.clone().detach()
            cnn1 = torch.load('best_model_new1.pt').cuda()
            seg, d1, d2, d3, d4 = cnn1(data_copy1, data_copy2)
            y = sig(seg)
            y = y.detach().cpu().numpy()
            y[y < 0.5] = 0
            label = label.detach().cpu().numpy()
            converted_image = (y * 255).astype(np.uint8)
            for k in range(2):
                t2, ypredict = cv2.threshold(converted_image[k][0], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ytrue = label[k].flatten().astype("int64")
                ypredict = ypredict.flatten()
                ytrue = ytrue.tolist()
                ypredict = ypredict.tolist()
                c = confusion_matrix(ytrue, ypredict)
                if c.shape == (2, 2):
                    confu1 += c
        TP = confu1[1][1]
        TN = confu1[0][0]
        FN = confu1[1][0]
        FP = confu1[0][1]
        confu1 = np.array([[0, 0], [0, 0]])
        if (TP + FN) != 0 and (TP + FP) != 0:
                iou = TP / (TP + FN + FP)
                crs.append(iou)
                print(iou)
                if crs[-1]>=max(crs):
                    torch.save(cnn, 'pt/best_model_test_LEVIR1.pt')
                    print(len(crs))


    num=np.array(NUM)
    loss=np.array(LOSS)
    plt.figure()
    plt.title('loss during training')  # 标题
    plt.plot(num, loss, label="train_loss")
    plt.legend()
    plt.grid()
    plt.show()
    print('end')
train()
