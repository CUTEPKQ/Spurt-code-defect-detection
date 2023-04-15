'''
Author: fyx
Date: 2023-04-10 18:22:37
LastEditTime: 2023-04-14 15:59:56
Description: 缺陷生成
'''
import cv2
import numpy as np
import os
import pandas as pd



class DataProcessor:
    def __init__(self,path:str= None,num:int= None) -> None:
        # 加载原始喷码图像
        self.path=path
        self.num=num

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
            

            




        
    

    def img_loss(self,picture_path:str= None,info_path:str= None)-> None:
        '''
        description: 
        ''' 

        
      
        
          
        pass

    
    def img_add(self)-> None:
        pass

  
    
    def img_shift(self) ->None:
        pass


def main()-> None:
    path="D:/Desktop/Destop_files/Spurt-code-defect-detection/data/1.png"
    img=cv2.imread(path)
    print(img)
    # processor=DataProcessor(path)
    # processor.img_shift(20,30)


if __name__ == '__main__':
    main()

