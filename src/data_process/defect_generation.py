'''
Author: fyx
Date: 2023-04-10 18:22:37
LastEditTime: 2023-04-24 21:07:06
Description: 缺陷生成
'''
import cv2
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split


def random_odd_number(a, b) -> int:
    num = np.random.randint(a, b+1)
    num = num if num % 2 == 1 else num + 1  # 如果生成的数为偶数，则加1变成奇数
    return num


class DataProcessor:
    def __init__(self,config:dict = None) -> None:
        # 加载原始喷码图像
        self.path=config['picture_path']
        self.num = config['num']
        self.save_path= config['save_path']
        self.lable_path = config['lable_path']
        self.config = config
        self.lables = []

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        lables = ['non-defect','defect']
        for i in range(len(lables)):
            self.label_generate(self.lable_path,lables[i]+'\r')
        
        


    def data_process(self)-> None:
        '''
        Author: fyx
        description: 对图片进行缺陷生成
        '''
        for id in range(1,self.num+1):
            #得到图片和信息路径
            path_name='picture_'+str(id)
            picture_name=os.path.join(self.path,path_name+'.jpg')
            self.lables.append(picture_name+'    '+str(0)+'\r')
            info_name=os.path.join(self.path,path_name+'.csv')
            self.img_loss(picture_path=picture_name,info_path=info_name,id=id)
            self.img_blur(picture_path=picture_name,id=id)
            self.img_add(picture_path=picture_name,id=id)
            self.add_noisy(picture_path=picture_name,id=id)
        
        train_list , other_list = train_test_split(self.lables,train_size=0.7)
        value_list , test_list  = train_test_split(other_list,train_size=0.6)

        for i in range(len(train_list)):
            self.label_generate(self.config['train_path'],train_list[i])
        
        for i in range(len(value_list)):
            self.label_generate(self.config['val_path'],value_list[i])

        for i in range(len(test_list)):
            self.label_generate(self.config['test_path'],test_list[i])


        print("successfully process pictures")
            


    def img_loss(self,picture_path:str= None,info_path:str= None,id:int= None)-> None:
        '''
        description: 处理缺陷
        ''' 
        img=cv2.imread(picture_path)
        img_pd=pd.read_csv(info_path)
        result=img.copy()
        mode=1
        #mode=1,单个字符漏印
        if mode==1:
            select=random.randint(0,img_pd.shape[0]-1)
            sub_pd=img_pd.iloc[select]
            print('缺失字符为',end='')
            print(sub_pd['info'])
            location=sub_pd.values[:-1]
            result[location[1]:location[1]+location[2],location[0]:location[0]+location[3],:]=[255,255,255]
            save_img_path=os.path.join(self.save_path,'loss1_picture'+str(id)+'.jpg')
            cv2.imwrite(save_img_path,result)
            #save lable
            self.lables.append(save_img_path+'    '+str(1)+'\r')
            print(f'successfully get loss1 form picture{id}')
        ## mode==2 缺失
        elif mode==2:
            h=random.randint(10,result.shape[0])
            start=random.randint(0,result.shape[0]-h)
            result[start:start+h]=[255,255,255]
            save_img_path=os.path.join(self.save_path,'loss2_picture'+str(id)+'.jpg')
            cv2.imwrite(save_img_path,result)
            self.lables.append(save_img_path+'    '+str(1)+'\r')
            print(f'successfully get loss2 form picture{id}')   
          
    def img_add(self,picture_path:str= None,id:int= None)-> None:
        # 读取图片
        img = cv2.imread(picture_path)
        # 创建一个空白掩膜，尺寸与原始图像相同
        # 产生污点的个数
        choice=np.random.choice([1,2,3],p=[0.7,0.2,0.1])
        for i in range(choice):
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            pollute_tpe=random.randint(1,2)
            if pollute_tpe==1:
                # 随机生成一个圆形区域
                radius=np.random.randint(5,10)
                center=(np.random.randint(radius,img.shape[1]-radius),
                        np.random.randint(radius,img.shape[0]-radius))
                cv2.circle(mask, center, radius, 255, -1)
            else :
                point1=[np.random.randint(5,img.shape[1]),np.random.randint(5,img.shape[0])]
                point2=[point1[0]+random.randint(5,10),point1[1]+random.randint(5,10)]
                cv2.rectangle(mask,point1,point2,255,-1)

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
            save_img_path=os.path.join(self.save_path,'pollute_picture'+str(id)+'.jpg')
            cv2.imwrite(save_img_path,result)
            self.lables.append(save_img_path+'    '+str(1)+'\r')
            print(f'successfully  pollute  picture{id}') 
    
    def add_noisy(self,picture_path:str= None,id:int= None)-> None:
        # Load the image
        image = cv2.imread(picture_path)
        mean = 0
        variance = 10
        sigma = np.sqrt(variance)
        noise = np.random.normal(mean, sigma, size=image.shape)

        # Add noise to the image
        noisy_image = cv2.add(image, noise.astype(np.uint8))

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image back to the original format
        result = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        save_img_path=os.path.join(self.save_path,'noisy_picture'+str(id)+'.jpg')
        cv2.imwrite(save_img_path,result)
        self.lables.append(save_img_path+'    '+str(1)+'\r')
        print(f'successfully  add noisy  with  picture{id}') 


    def img_blur(self,picture_path:str= None,id:int= None)-> None:
        # 读取图片
        img = cv2.imread(picture_path)
        noise = np.zeros(img.shape, np.uint8)
        cv2.randn(noise, 0, 50)
        noisy_img = cv2.add(img, noise)
        kernel_size=np.random.choice([i for i in range(7,20,2)])
        # 进行高斯滤波
        result= cv2.GaussianBlur(noisy_img, (kernel_size,kernel_size),5)
        save_img_path=os.path.join(self.save_path,'blur_picture'+str(id)+'.jpg')
        cv2.imwrite(save_img_path,result)
        self.lables.append(save_img_path+'    '+str(1)+'\r')
        print(f'successfully  blur  picture{id}') 
    

    def label_generate(self,path:str = None,data:str = None) -> None:
        write_obj=open(path,'a')
        with write_obj as f:
            f.write(data)
        


def main()-> None:
    config={}
    config['picture_path'] = os.path.join(os.path.dirname(__file__),'..\..', 'data','non-defect') 
    config['save_path'] = os.path.join(os.path.dirname(__file__), '..\..','data','defect') 
    config['lable_path'] = os.path.join(os.path.dirname(__file__), '..\..','data','lables.txt') 
    config['train_path'] = os.path.join(os.path.dirname(__file__), '..\..','data','train_list.txt') 
    config['test_path'] = os.path.join(os.path.dirname(__file__), '..\..','data','test_list.txt')
    config['val_path'] = os.path.join(os.path.dirname(__file__), '..\..','data','val_list.txt')
    config['num'] = int(200)
    processor=DataProcessor(config=config)
    processor.data_process()
    # processor.img_shift(20,30)


if __name__ == '__main__':
    main()

