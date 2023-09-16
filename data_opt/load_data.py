
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
import pandas as pd
import matplotlib as mpl
import random
mpl.use('Agg')

import os
import glob
from util.misc import NestedTensor
from config import *

from .heat_map import HeatMap
class VideoAttTarget_image(Dataset):
    def __init__(self, annotation_dir, transform,args,
                 max_label_size=maxlabeledbboxs):

        annotation_file =[f for f in  glob.glob(os.path.join(annotation_dir, '*')) if f.endswith(".csv") ][0]

        self.df = pd.read_csv(os.path.join(annotation_dir,annotation_file))

        self.data_dir = videoattentiontarget_data
        self.transform = transform
        self.max_label_size=max_label_size

        self.length = self.df.shape[0]
        self.heat_map=HeatMap(*args.heat_dim)
        self.heat_map_shape=args.heat_dim[:2]

        self.targets_indexs=["bboxs","heads","inouts","heat_maps","labels"]
    def __getitem__(self, index):
        row = self.df.iloc[index]

        data = eval(row['data'])
        choice_data=random.choice(data)
        path = choice_data["path"]
        rows=choice_data["label"]

        # read image
        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        width, height = img.size

        bboxs=[]
        heads=[] # 0  False;1 True
        inouts=[] # 0 out ;1 in
        heat_maps=[]

        img_flip=False
        # Random flip
        if np.random.random_sample() <= 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip=True
        # Random color change
        if np.random.random_sample() <= 0.5:
            img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))

        if self.transform is not None:
            img = self.transform(img)

        for row in rows:
            x_min = row['xmin']
            y_min = row['ymin']
            x_max = row['xmax']
            y_max = row['ymax']
            gaze_x = row['gazex']
            gaze_y = row['gazey']

            if gaze_x == -1 and gaze_y == -1:
                gaze_inside = 0
            else:
                if gaze_x < 0:  # move gaze point that was sliglty outside the image back in
                    gaze_x = 0
                if gaze_y < 0:
                    gaze_y = 0
                gaze_inside = 1


            # gaze_x = gaze_x / width
            # gaze_y = gaze_y / height

            # expand face bbox a bit
            k = 0.1
            x_min -= k * abs(x_max - x_min)
            y_min -= k * abs(y_max - y_min)
            x_max += k * abs(x_max - x_min)
            y_max += k * abs(y_max - y_min)
            x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

            if img_flip:
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = width - gaze_x

            gaze_heatmap=self.heat_map(width, height,gaze_x,gaze_y) if gaze_inside else torch.zeros(*self.heat_map_shape)
            gaze_heatmap=gaze_heatmap.view(1,-1)
            x_min,x_max=x_min/width,x_max/width
            y_min,y_max=y_min/height,y_max/height

            bbox=torch.tensor([x_min,y_min,x_max,y_max]).view(1,-1)
            head=torch.tensor([1])
            inout=torch.tensor([gaze_inside])

            bboxs.append(bbox)
            heads.append(head)
            inouts.append(inout)
            heat_maps.append(gaze_heatmap)
        label=[torch.ones(1,dtype=torch.long) for _ in range(len(heads))]

        bboxs,heads,inouts,heat_maps,label,mask=self.padding_label(bboxs,heads,inouts,heat_maps,label)

        return img,bboxs,heads,inouts,heat_maps,label,mask

    def padding_label(self,bboxs,heads,inouts,heat_maps,label):
        n_size=len(bboxs)
        mask=torch.ones(7,7,dtype=torch.int64,)
        # mask[:,:]=True
        for i in range(self.max_label_size-n_size):
            bboxs.append(torch.zeros(1,4))
            heads.append(torch.zeros(1))
            inouts.append(torch.zeros(1))
            label.append(torch.zeros(1,dtype=torch.long))
            heat_maps.append(torch.zeros(1,self.heat_map_shape[0]*self.heat_map_shape[1]))
        return bboxs,heads,inouts,heat_maps,label,mask

    def __len__(self):
        return self.length

    def collection_function(self,data):
        img=torch.stack([v[0] for v in data])
        mask=torch.stack([v[-1] for v in data])
        targets=[{key:torch.cat(value,dim=0) if key not in ["heads","inouts"] else torch.cat(value,dim=0).long() for key,value in zip(self.targets_indexs,v[1:-1])} for v in data]
        return img,targets,mask


class PredictPicture():
    def __init__(self,model,transform,args):
        self.model=model
        self.transform=transform
        self.heat_with=args.heat_dim[0]
        self.heat_height=args.heat_dim[1]

    def __call__(self,picture_path):

        # read image
        img = Image.open(picture_path)
        img = img.convert('RGB')
        width, height = img.size

        if self.transform is not None:
            img = self.transform(img)

        mask = torch.ones(1,7, 7, dtype=torch.int64, )
        with torch.no_grad():
            outputs = self.model(NestedTensor(img.unsqueeze(0), mask))

        return self.rec(outputs,width,height)

    def rec(self,predict,width,height):
        predict={key:value.squeeze() for key,value in predict.items()}
        heads=torch.argmax(predict["pred_head"],dim=-1)==1
        boxes=torch.einsum('ik,k -> ik',predict["pred_boxes"],torch.tensor([width,height,width,height]))

        boxes=boxes[heads]
        watch_out=torch.argmax(predict["pred_watchout"][heads],dim=-1)==1
        heatmap=predict["pred_heatmap"][heads].view(-1,self.heat_with,self.heat_height)

        result=[{"box":[int(i) for i in a.tolist()],"watchinside":1 if b else 0 ,"heatmap":c.tolist()  } for a,b,c in zip(boxes,watch_out,heatmap)]
        return result

