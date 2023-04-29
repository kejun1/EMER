import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import re
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import pandas as pd

def Split(imglist,n):
    k, m=divmod(len(imglist),n)
    return [imglist[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in list(range(n))]


class My_FEV_Dataset(Dataset):

    def __init__(self, v_root_dir, e_root_dir,k_test, transform=None, is_train=True):
        self.v_root_dir = v_root_dir
        self.v_path = os.path.join(self.v_root_dir)# D:\mydataset\train\face
        self.img_path = []
        if is_train:
            for i in range(1,6):
                if i == k_test:
                    continue
                set = os.listdir(os.path.join(self.v_path,"set"+str(i)))
                f_set = []
                for iter in set:
                    i_str = "set"+str(i)+"/"+iter
                    f_set.append(i_str)
                self.img_path += f_set
        else:
            set = os.listdir(os.path.join(self.v_path, "set"+str(k_test)))
            f_set = []
            for iter in set:
                i_str = "set" + str(k_test) + "/" + iter
                f_set.append(i_str)
            self.img_path += f_set

        # self.img_path.sort(key=lambda x: int(re.split(r'[_ .()]', x)[0]))
        self.e_root_dir = e_root_dir
        # self.e_path = os.path.join(self.e_root_dir)  # D:\mydataset\train\eye
        # self.eye_path = os.listdir(self.e_path)  # 3_film_happy Happy
        # self.eye_path.sort(key=lambda x: int(re.split(r'[_ .]', x)[0]))
        self.transform=transform
        self.label = pd.read_excel(r'D:/SMAD/data/smead_cross/new_label.xlsx')

    def __getitem__(self, item):
        img_name = self.img_path[item]# 3_film_happy Happy
        img_path = os.path.join(self.v_root_dir,img_name)# D:\mydataset\train\face\3_film_happy Happy
        # img_path = self.v_root_dir+'//'+img_name
        # print(img_name)
        # print(img_path)
        iter_img = re.split(r'[\\/]', img_path)[-1]
        eye_name = iter_img+".xlsx"
        eye_path = os.path.join(self.e_root_dir, eye_name)
        # print(eye_path)
        eyedata = pd.read_excel(eye_path)
        try:
            videodata = eyedata[['Gaze point X','Gaze point Y']]
        except:
            videodata = eyedata[['Gaze point X [DACS px]', 'Gaze point Y [DACS px]']]
        try:
            eyedata = np.array(eyedata)
            list_split = Split(eyedata, 32)
            eyelist = [random.choice(i) for i in list_split]
            eyelist = np.array(eyelist).astype(float)
            eyedata = torch.tensor(eyelist)
            eyedata = eyedata.type(torch.float32)
        except:
            print(eye_path)

        videodata = np.array(videodata)
        list_split = Split(videodata, 32)
        videolist = [random.choice(i) for i in list_split]
        videolist = np.array(videolist)
        videodata = torch.tensor(videolist)
        videodata = videodata.type(torch.float32)

        imglist=[]
        filenames = os.listdir(img_path)


        filenames.sort(key=lambda x: x[-5])
        for filename in filenames:
            img_item_path = os.path.join(img_path, filename)
            img = cv2.imread(img_item_path)


            img = cv2.resize(img, (112, 112))
            if self.transform is not None:
                img = self.transform(img)

            imglist.append(img)
        list_split = Split(imglist, 8)
        imglist = [random.choice(i) for i in list_split]
        face = np.stack(imglist, axis=0)
        e_index = self.label[self.label['ID'] == iter_img].index.tolist()[0]
        label_emotion = self.label.loc[e_index, 'Emotion']
        label_valence = self.label.loc[e_index, 'Valence']
        label_arousal = self.label.loc[e_index, 'Arousal']
        label_emotion = torch.as_tensor(np.array(label_emotion))
        label_valence = torch.as_tensor(np.array(label_valence))
        label_arousal = torch.as_tensor(np.array(label_arousal))
        return face, eyedata, videodata, label_emotion, label_valence, label_arousal


    def __len__(self):
        return len(self.img_path)

def Load_FEV_Dataset(v_train_path,e_train_path, v_test_path,e_test_path,k_test,batch_size, num_workers=2):
    transform_train = transforms.Compose([

        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = My_FEV_Dataset(v_train_path, e_train_path, k_test, transform=transform_train, is_train=True)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = My_FEV_Dataset(v_test_path,e_test_path, k_test, transform=transform_test, is_train=False)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("数据集加载完成!!!")

    return trainloader, testloader

# if __name__ == '__main__':
#     v_train_path = r'F:\data_final\face_light_align'
#     e_train_path = r'F:\data_final\final_eye'
#     trainset = My_FEV_Dataset(v_train_path, e_train_path,transform=None)
#
#     for data in trainset:
#         face,eyedata,videodata,label_emotion,label_valence,label_arousal=data
#         print(label_valence)
