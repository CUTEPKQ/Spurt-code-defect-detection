'''
Author: fengyx ,fengyx@hnu.edu.cn
Date: 2023-05-11 19:38:20
LastEditors: fengyx ,fengyx@hnu.edu.cn
LastEditTime: 2023-05-15 18:34:53
Description: 
'''
import os
import paddlex as pdx
import cv2

from opt import get_opts
from paddlex import transforms as T


class MyModel:
    def __init__(self,hparams) -> None:
        self.hparams = hparams
        self.model = pdx.cls.MobileNetV3_small(num_classes = self.hparams.num_classes)

    def train(self,
              train_dataset,
              eval_dataset,
              lr_decay_epochs) -> None:

        self.model.train(num_epochs = self.hparams.num_epochs,                       
                          train_dataset = train_dataset,
                          train_batch_size = self.hparams.train_batch_size,
                          eval_dataset = eval_dataset,
                          lr_decay_epochs = lr_decay_epochs,
                          learning_rate = 0.01,
                          save_dir = self.hparams.save_dir,
                          use_vdl = self.hparams.ues_vld)

    def load_model(self,path) -> None:
        self.model = pdx.load_model(path)
    
    def predict(self,img) :
        return self.model.predict(img)
    
    def test(self,test_data,com_result):
        results = self.model.predict(test_data)
        predict = []
        for result in results:
            if 'category_id' in result[0].keys():
                category_id = result[0]['category_id']
                predict.append(category_id)
        p = 0
        for i in range(len(com_result)):
            if(com_result[i]==predict[i]):
                p = p+1
        print("the accuracy for test data is: ",p/len(predict))


def data_handle() -> None:
    handle_list = ['train_list.txt','test_list.txt','val_list.txt']
    path = "data"
    test_data = []
    com_result = []
    
    datanames = os.listdir(path)
    for i in datanames:
        if i in handle_list:
            new_lines = []
            with open("data/" + i, "r") as f:
                lines = f.readlines()  
                for line in lines:
                    if line.startswith(".."):
                        new_line = line[11:]
                        new_lines.append(new_line)
                        if i =='test_list.txt':
                            test_data.append(cv2.imread(new_line))
                            com_result.append(int(line[-2]))
                    else :
                        new_lines.append(line)

            with open("data/" + i, "w") as f:
                for new_line in new_lines:
                    f.writelines(new_line)
    return test_data,com_result

if __name__=='__main__':
    # get hparams
    hparams = get_opts()
    # get test_data and table
    test_data,com_result = data_handle()
    # data augmention
    train_transforms = T.Compose(
    [T.RandomCrop(crop_size=224), T.RandomHorizontalFlip(), T.Normalize()])
    eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256), T.CenterCrop(crop_size=224), T.Normalize()])
    data_dir = '/home/aistudio/Spurt-code-defect-detection/data'
    file_train = data_dir+'train_list.txt'
    flie_val = data_dir+'val_list.txt'
    label_list = data_dir+'labels.txt'

    train_dataset = pdx.datasets.ImageNet(
        data_dir='data',
        file_list='data/train_list.txt',
        label_list='data/labels.txt',
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.ImageNet(
        data_dir='data',
        file_list='data/val_list.txt',
        label_list='data/labels.txt',
        transforms=eval_transforms)

    mymodel = MyModel(hparams)
    mymodel.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8])
    # model_path = ...
    # mymodel.load_model(model_path)
    mymodel.test(test_data,com_result)
    