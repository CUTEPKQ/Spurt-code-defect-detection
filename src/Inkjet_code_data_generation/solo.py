from PIL import Image, ImageDraw, ImageFont
import os
import random
import csv


# 加载字体文件
font_path = os.path.join(os.getcwd(), 'ttf/point.ttf')


# 英文字典
import string
en_dic = string.printable
en_dic_list = list(en_dic)
en_dic_list = en_dic_list[0:62]

# 背景
background_color = (255, 255, 255)


# 保存位置
save_folder = 'data/'
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
    # 创建需要的字符串
    for i in range(num):
        random_char = random.choice(en_dic_list)
        text += random_char
    return text



for k in range(10):

    # 确认字体
    font_size = random.randint(20, 40)
    font = ImageFont.truetype(font_path, font_size)

    # 确认文本
    text = create_text()

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
    text_color = (random.randint(20, 255), random.randint(20, 255), random.randint(20, 255))
    origin_x = int(random.random()*(w - origin_w))
    origin_y = int(random.random()*(h - origin_h))
    output_size = (origin_x, origin_y)
    draw.text(output_size, text, text_color, font=font)



    # 确定保存路径
    dicname = dicname_format.format(k + 1)
    picture_filename = picture_filename_format.format(k + 1)
    csv_filename = csv_filename_format.format(k + 1)
    save_path = save_folder + dicname



    # 起始
    x = origin_x
    # 保存text中每个char的位置信息
    char_positions = []
    for char in text:
        # 字符宽高
        char_width = font.getsize(char)[0]
        char_height = font.getsize(char)[1]

        char_bbox = (x, origin_y, char_height, char_width, char)
        char_positions.append(char_bbox)
        x += char_width

    print(text, char_positions)




    if not os.path.exists(save_path):
        os.makedirs(save_path)



    for bbox in char_positions:
        draw.rectangle((bbox[0],bbox[1],bbox[0]+bbox[3], bbox[1]+bbox[2]) , outline="black")

    # bbox = font.getbbox(text)
    # print('bbox' , bbox)
    # draw.rectangle(bbox, outline="black")

    image.save(save_path + picture_filename, 'JPEG')



    with open(save_path+csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['x', 'y', 'h', 'w', 'info'])
        for bbox in char_positions:
            x, y, h, w ,info  = bbox
            writer.writerow([x, y, h, w, info])

