from PIL import Image, ImageDraw, ImageFont
import os
import random
import csv
import string

# 英文字典
# 数字，英文大小写
en_dic = string.printable
# print(en_dic)

en_dic_list = list(en_dic)
en_dic_list = en_dic_list[0:62]


# 背景
background_color = (255, 255, 255)

current_path=os.path.dirname(__file__)
# 加载字体文件
font_path = os.path.join(current_path, 'ttf/point.ttf')
# 保存位置
save_folder =os.path.join(current_path,'../..','data/','non-defect/') 
dicname_format = 'picture_{}/'
picture_filename_format = 'picture_{}.jpg'
csv_filename_format = 'picture_{}.csv'
text_file = 'data/train_m.txt'


def create_text():
    """
    # 个数数随机(1,17)
     字符随机
    :return: text
    """
    num = random.randint(1, 17)
    text = ''
    text_idx = []
    # 创建需要的字符串
    for i in range(num):
        random_char = random.choice(en_dic_list)
        text += random_char
        text_idx.append(i)
    return text,text_idx




def get_pos_rate_dic(text,font_size):
    # text = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    font = ImageFont.truetype(font_path, font_size)

    # 初始文本框
    bbox = font.getbbox(text)

    origin_w = bbox[2]
    origin_h = bbox[3]

    image_size = (origin_w, origin_h)
    image = Image.new('RGB', image_size, background_color)
    draw = ImageDraw.Draw(image)

    image = Image.new('RGB', image_size, background_color)
    # 背景
    draw = ImageDraw.Draw(image)

    # word
    draw.text([0,0], text, 0 , font=font)

    x = 0
    origin_y =0
    # 保存text中每个char的位置信息
    char_positions = []
    for char in text:
        # 字符宽高
        char_width = font.getsize(char)[0]
        char_height = font.getsize(char)[1]

        char_bbox = (x, origin_y, char_height, char_width, char)
        char_positions.append(char_bbox)
        x += char_width

    # 标准位置
    # for bbox in char_positions:
    #     draw.rectangle((bbox[0], bbox[1], bbox[0] + bbox[3], bbox[1] + bbox[2]), outline="black")


    # 创建对应比例的映射
    h_rate_list = []
    w_rate_list = []


    for bbox in char_positions:
        w_rate=0
        # 从右往左，get  w
        flag = 0
        for i in range( bbox[0]-1 + bbox[3] , bbox[0] , -1):
            for j in range( bbox[1]-1 ,bbox[1] + bbox[2]-1):
                if image.getpixel((i,j)) !=(255,255,255):
                    # print('i: ',i, 'bbox[0]:' , bbox[0], '3: ' ,bbox[3],'res ;',w_rate)
                    w_rate =(i-bbox[0]) / bbox[3]
                    w_rate_list.append(w_rate)
                    flag = 1
                    break
            if flag:
                break



    for bbox in char_positions:
        h_rate = 0
        flag = 0
        for j in range(bbox[1]+1, bbox[1] + bbox[2]):
            for i in range( bbox[0] + bbox[3] -1, bbox[0] , -1):
                if image.getpixel((i,j)) !=(255,255,255):
                    h_rate =(bbox[1] + bbox[2]-j) / bbox[2]
                    h_rate_list.append(h_rate)
                    flag = 1
                    break
            if flag:
                break



    # print((h_rate_list))
    # image.show()
    return h_rate_list,w_rate_list





def create_data(picture_num):



    for k in range(picture_num):

        # 确认字体
        font_size = random.randint(20, 70)
        font = ImageFont.truetype(font_path, font_size)

        # 确认文本,和序列
        text,text_idx = create_text()
        h_rate_list, w_rate_list = get_pos_rate_dic(text, 40)

        # print(font_size)
        # text = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # text_idx = [i for i in range(0,62)]
        # print(text_idx)

        # 初始文本框
        bbox = font.getbbox(text)
        # print('bbox' , bbox)
        origin_w = bbox[2]
        origin_h = bbox[3]
        # print('origin', origin_w , origin_h)

        # 创建图像
        h = int((1 + random.random()) * origin_h)
        w = int((1 + random.random()) * origin_w)
        # print('img', w, h)
        image_size = (w, h)

        # 背景绘制
        image = Image.new('RGB', image_size, background_color)
        draw = ImageDraw.Draw(image)

        # 文字位置 & 绘制
        # output_size = (1, 1)
        # draw.text(output_size, text, text_color, font=font)
        text_color = (0, 0, 0)
        origin_x = int(random.random() * (w - origin_w))
        origin_y = int(random.random() * (h - origin_h))
        output_size = (origin_x, origin_y)
        draw.text(output_size, text, text_color, font=font)

        # 确定保存路径
        dicname = dicname_format.format(k + 1)
        picture_filename = picture_filename_format.format(k + 1)
        csv_filename = csv_filename_format.format(k + 1)
        save_path = save_folder 


        # 起始
        x = origin_x
        # 保存text中每个char的位置信息
        char_positions = []


        for char,idx in zip(text,text_idx):
            # 字符宽高
            char_width = font.getsize(char)[0]
            char_height = font.getsize(char)[1]

            # new y
            y = origin_y
            y = int((y+char_height) - char_height * h_rate_list[idx]) -1

            # new height
            h = int(char_height * h_rate_list[idx]) + 1

            # new width
            w = int(char_width * w_rate_list[idx]) + 1
            # print(char,x)

            char_bbox = (x, y, h, w, char)
            char_positions.append(char_bbox)

            x += char_width

        # print(text, char_positions)



        if not os.path.exists(save_path):
            os.makedirs(save_path)



        # for bbox in char_positions:
        #     draw.rectangle((bbox[0], bbox[1], bbox[0] + bbox[3], bbox[1] + bbox[2]), outline="black")

        # bbox = font.getbbox(text)
        # print('bbox' , bbox)
        # draw.rectangle(bbox, outline="black")

        image.save(save_path + picture_filename, 'JPEG')

        with open(save_path + csv_filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['x', 'y', 'h', 'w', 'info'])
            for bbox in char_positions:
                x, y, h, w, info = bbox
                writer.writerow([x, y, h, w, info])


create_data(200)


# get_pos_rate_dic(20)