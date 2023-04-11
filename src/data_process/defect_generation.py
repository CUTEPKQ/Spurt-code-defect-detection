'''
Author: fyx
Date: 2023-04-10 18:22:37
LastEditTime: 2023-04-10 21:06:05
Description: 缺陷生成
'''
import cv2
import numpy as np

class DataProcessor:
    def __init__(self,path:str= None) -> None:
        # 加载原始喷码图像
        self.img=cv2.imread(path)
        self.radius=1

    def draw_circle(self,event, x, y,flags,param)-> None:

        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(self.img, (x, y), self.radius, (0,0,0), -1)
 
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.circle(self.img, (x, y), self.radius, (0,0,0), -1)
 

    def draw_rectangle(self,event, x, y,flags,param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
             # 记录矩形的起始点坐标
            param["start_point"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            #cv2.rectangle(self.img, param["start_point"], (x, y), (0, 0, 255), 2)
            cv2.rectangle(self.img, param["start_point"], (x, y), (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            # 记录矩形的结束点坐标
            param["end_point"] = (x, y)
            # x1, y1 = param["start_point"]
            # x2, y2 = param["end_point"]
            # x, y = min(x1, x2), min(y1, y2)
            # w, h = abs(x2 - x1), abs(y2 - y1)
            cv2.rectangle(self.img, param["start_point"], param["end_point"], (255, 255, 255), -1)
        
    def img_show(self, img_result:np= None)-> None:
        cv2.imshow('result', img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def img_loss(self)-> None:
        # 显示原始图像，并设置鼠标回调函数
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_rectangle,{"start_point": None, "end_point": None})
        # 循环显示图像，直到按下ESC键退出
        while True:
            cv2.imshow("image", self.img)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.imwrite("result.jpg", self.img)
                break
    
    def img_add(self)-> None:
        # 显示原始图像，并设置鼠标回调函数
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_circle)
        while True:
            cv2.imshow('image', self.img)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.imwrite("result.jpg", self.img)
                break
        # 将多余的喷墨应用于原始图像
  
    
    def img_shift(self,dx:int= None,dy:int= None) ->None:
        # 定义仿射变换矩阵
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        # 应用仿射变换
        result = cv2.warpAffine(self.img, M, (self.img.shape[1], self.img.shape[0]))

        self.img_show(result)
        


def main()-> None:
    path="D:/Desktop/Destop_files/Spurt-code-defect-detection/data/2.jpg"
    processor=DataProcessor(path)
    processor.img_shift(40,60)


if __name__ == '__main__':
    main()

