'''
Author: fyx
Date: 2023-04-10 18:22:37
LastEditTime: 2023-04-18 19:44:11
Description: 缺陷生成
'''
import cv2
import numpy as np
import os
import pandas as pd
import random

def random_odd_number(a, b) -> int:
    num = np.random.randint(a, b+1)
    num = num if num % 2 == 1 else num + 1  # 如果生成的数为偶数，则加1变成奇数
    return num


class DataProcessor:
    def __init__(self,path:str= None,num:int= None,save_path:str= None) -> None:
        # 加载原始喷码图像
        self.path=path
        self.num=num
        self.save_path=save_path
        

    def data_process(self)-> None:
        '''
        Author: fyx
        description: 对图片进行缺陷生成
        '''
        for id in range(1,self.num+1):
            #得到图片和信息路径
            path_name='picture_'+str(id)
            file_path=os.path.join(self.path,path_name)
            picture_name=os.path.join(file_path,path_name+'.jpg')
            print(picture_name)
            info_name=os.path.join(file_path,path_name+'.csv')
            self.img_loss(picture_path=picture_name,info_path=info_name,id=id)
            self.img_blur(picture_path=picture_name,id=id)
            self.img_add(picture_path=picture_name,id=id)
        
        print("successfully process pictures")
            


    def img_loss(self,picture_path:str= None,info_path:str= None,id:int= None)-> None:
        '''
        description: 处理缺陷
        ''' 
        img=cv2.imread(picture_path)
        img_pd=pd.read_csv(info_path)
        result=img.copy()
        save_img_fold=os.path.join(self.save_path,'picture'+str(id))
        if not os.path.exists(save_img_fold):
            os.makedirs(save_img_fold)
        mode=np.random.randint(1,2)
        #mode=1,单个字符漏印
        if mode==1:
            select=random.randint(0,img_pd.shape[0]-1)
            sub_pd=img_pd.iloc[select]
            print('缺失字符为',end='')
            print(sub_pd['info'])
            location=sub_pd.values[:-1]
            result[location[1]:location[1]+location[2],location[0]:location[0]+location[3],:]=[255,255,255]
            save_img_path=os.path.join(save_img_fold,'loss1_picture'+str(id)+'.jpg')
            cv2.imwrite(save_img_path,result)
            print(f'successfully get loss1 form picture{id}')
        ## mode==2 缺失
        elif mode==2:
            h=random.randint(10,result.shape[0])
            start=random.randint(0,result.shape[0]-h)
            result[start:start+h]=[255,255,255]
            save_img_path=os.path.join(save_img_fold,'loss2_picture'+str(id)+'.jpg')
            cv2.imwrite(save_img_path,result)
            print(f'successfully get loss2 form picture{id}')   
          
    def img_add(self,picture_path:str= None,id:int= None)-> None:
        save_img_fold=os.path.join(self.save_path,'picture'+str(id))
        if not os.path.exists(save_img_fold):
            os.makedirs(save_img_fold)

        # 读取图片
        img = cv2.imread(picture_path)
        # 创建一个空白掩膜，尺寸与原始图像相同
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # 随机生成一个圆形区域
        radius=np.random.randint(10,20)
        center=(np.random.randint(radius,img.shape[1]-radius),
                np.random.randint(radius,img.shape[0]-radius))
        cv2.circle(mask, center, radius, 255, -1)
        # 随机生成内核的形状、大小和方向
        kernel_shape = np.random.choice([cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE])
        kernel_size = np.random.randint(3, 10)
        kernel_rotation = np.random.randint(0, 180)
        # 创建随机形状的内核
        kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), 
        kernel_rotation, 1.0), (kernel_size, kernel_size))
        # 对原始图像进行膨胀操作，重复多次以增加目标物体的面积
        iterations = 3
        mask = cv2.dilate(mask, kernel, iterations=iterations)
        # 将掩膜反转，变成黑色圆形区域的掩膜
        mask = cv2.bitwise_not(mask)
        # 将原始图像和掩膜进行与操作，保留圆形区域以外的颜色
        result = cv2.bitwise_and(img, img, mask=mask)
        save_img_path=os.path.join(save_img_fold,'loss2_picture'+str(id)+'.jpg')
        cv2.imwrite(save_img_path,result)
        print(f'successfully  pollute  picture{id}') 
  
    def img_blur(self,picture_path:str= None,id:int= None)-> None:
        save_img_fold=os.path.join(self.save_path,'picture'+str(id))
        if not os.path.exists(save_img_fold):
            os.makedirs(save_img_fold)
        # 读取图片
        img = cv2.imread(picture_path)
        noise = np.zeros(img.shape, np.uint8)
        cv2.randn(noise, 0, 50)
        noisy_img = cv2.add(img, noise)
        kernel_size=np.random.choice([i for i in range(7,20,2)])
        # 进行高斯滤波
        result= cv2.GaussianBlur(noisy_img, (kernel_size,kernel_size),5)
        save_img_path=os.path.join(save_img_fold,'loss2_picture'+str(id)+'.jpg')
        cv2.imwrite(save_img_path,result)
        print(f'successfully  blur  picture{id}') 


def main()-> None:
    picture_path="D:/Desktop/Destop_files/Spurt-code-defect-detection/data/origin_data/"
    save_path="D:/Desktop/Destop_files/Spurt-code-defect-detection/data/process_data/"
    processor=DataProcessor(picture_path,10,save_path)
    processor.data_process()
    # processor.img_shift(20,30)


if __name__ == '__main__':
    main()

