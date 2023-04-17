'''
Author: fyx
Date: 2023-04-10 18:22:37
LastEditTime: 2023-04-17 17:45:18
Description: 缺陷生成
'''
import cv2
import numpy as np
import os
import pandas as pd
import random


class DataProcessor:
    def __init__(self,path:str= None,num:int= None,save_path:str= None) -> None:
        # 加载原始喷码图像
        self.path=path
        self.num=num
        self.save_path=save_path
        

    def data_process(self,img_path:str= None,info_path:str= None)-> None:
        '''
        Author: fyx
        description: 对图片进行缺陷生成
        '''
        for id in range(self.num):
            #得到图片和信息路径
            path_name='picture_'+str(id)
            flie_path=os.path.join(self.path,path_name)
            picture_name=path_name+'.jpg'
            info_name=path_name+'.csv'
            
            


    def img_loss(self,picture_path:str= None,info_path:str= None,id:int= None)-> None:
        '''
        description: 处理缺陷
        ''' 
        img=cv2.imread(picture_path)
        print(img.shape)
        img_pd=pd.read_csv(info_path)
        select=random.randint(0,img_pd.shape[0]-1)
        sub_pd=img_pd.iloc[select]
        location=sub_pd.values[:-1]
        result=img.copy()
        save_img_fold=os.path.join(self.save_path,'processed_data','picture'+str(id))
        if not os.path.exists(save_img_fold):
            os.makedirs(save_img_fold)
        mode=1
        #mode=1,单个字符漏印
        if mode==1:
            result[location[0]:location[0]+location[2],location[1]:location[1]+location[3],:]=[255,255,255]
            save_img_path=os.path.join(save_img_fold,'loss1_picture'+str(id)+'.jpg')
            cv2.imwrite(save_img_path,result)
            print(f'successfully get loss1 form picture{id}')
        ## mode==2 缺失
        elif mode==2:
            h=random.randint(0,result.shape[0])
            start=random.randint(0,result.shape[0]-h)
            result[start:start+h, :,]=[255,255,255]
            save_img_path=os.path.join(save_img_fold,'loss2_picture'+str(id)+'.jpg')
            cv2.imwrite(save_img_path,result)
            print(f'successfully get loss2 form picture{id}')   
          
    def img_add(self,picture_path:str= None,info_path:str= None,id:int= None)-> None:
        img=cv2.imread(picture_path)
        img_info=pd.read_csv(info_path)
        # 随机生成污染区域的位置和大小
        h, w = img.shape[:2]
        x = np.random.randint(0, w-100)
        y = np.random.randint(0, h-100)
        width = np.random.randint(50, 100)
        height = np.random.randint(50, 100)

        # 随机生成污染区域的形状
        num_points = np.random.randint(5, 20)
        points = np.zeros((num_points, 2), dtype=np.int32)
        for i in range(num_points):
            points[i, 0] = np.random.randint(x, x+width)
            points[i, 1] = np.random.randint(y, y+height)

        # 创建掩模图像
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [points], (255, 255, 255))

        # 将掩模应用于原图像
        result = cv2.bitwise_and(img, mask)

        pass

  
    
    def img_shift(self) ->None:
        pass


def main()-> None:
    path3='D:/Desktop/Destop_files/Spurt-code-defect-detection/data/'
    path1="D:/Desktop/Destop_files/Spurt-code-defect-detection/data/origin_data/picture_1/picture_1.jpg"
    path2="D:/Desktop/Destop_files/Spurt-code-defect-detection/data/origin_data/picture_1/picture_1.csv"
    processor=DataProcessor(path1,2,path3)
    processor.img_loss(path1,path2,1)
    # processor.img_shift(20,30)


if __name__ == '__main__':
    main()

