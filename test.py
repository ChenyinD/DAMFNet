import torch
import datasets.dataset as dataset
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2




def train():
    # train
    # 超参数
    sig=nn.Sigmoid()
    EPOCH =1 # 前向后向传播迭代次数


    # f2 = dataset.load_dataset_224(2, '../DAFNet/datasets/txt/LEVIR-CD/A3.txt', '../DAFNet/datasets/txt/LEVIR-CD/B3.txt', '../DAFNet/datasets/txt/LEVIR-CD/C3.txt')
    f2 = dataset.load_dataset_224(2, '../DAFNet/datasets/txt/WHU-CD/WA1.txt', '../DAFNet/datasets/txt/WHU-CD/WB1.txt', '../DAFNet/datasets/txt/WHU-CD/WC1.txt')


    cnn = torch.load('../DAFNet/pt/best_model_test_WHU.pt').cuda()

    confu=np.array([[0, 0], [0, 0]])
    for epoch in range(EPOCH):
        for step, (images, images2, label, filename) in enumerate(f2):
            images = images.cuda()
            images = (images - torch.min(images)) / (torch.max(images) - torch.min(images))
            data_copy1 = images.clone().detach()
            images2 = images2.cuda()
            images2 = (images2 - torch.min(images2)) / (torch.max(images2) - torch.min(images2))
            data_copy2 = images2.clone().detach()
            labels = label.cuda()
            seg,d1,d2,d3,d4=cnn(data_copy1,data_copy2)
            label = labels.detach().cpu().numpy()
            y = sig(seg)
            y = y.detach().cpu().numpy()
            y[y < 0.5] = 0
            converted_image = (y * 255).astype(np.uint8)
            for j in range(2):
             t2, ypredict = cv2.threshold(converted_image[j][0], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

             ytrue = label[j].flatten().astype("int64")
             ypredict1 = ypredict.flatten()
             ytrue = ytrue.tolist()
             ypredict1 = ypredict1.tolist()
             c = confusion_matrix(ytrue, ypredict1)
             if c.shape ==(2,2):
              confu += c
        TP = confu[1][1]
        TN = confu[0][0]
        FN = confu[1][0]
        FP = confu[0][1]
        confu = np.array([[0, 0], [0, 0]])
        if (TP + FN) != 0 and (TP + FP) != 0:
            oa = (TP + TN) / (TP + TN + FP + FN)
            recall = (TP) / (TP + FN)
            presion = TP / (TP + FP)
            iou = TP / (TP + FN + FP)
            f = 2 * recall * presion / (recall + presion)
            print(oa,presion, recall, iou,f, 'o精召差f')
train()