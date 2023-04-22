from lxml.etree import Element, SubElement, tostring
from utils import log
import shutil
import json
import os


def get_pic_dir(out_put_dir):
    img_dir = os.path.join(out_put_dir, "img")
    pic_dir = os.path.join(img_dir, "pic")
    return pic_dir


def get_fragment_dir(out_put_dir):
    img_dir = os.path.join(out_put_dir, "img")
    fragment_dir = os.path.join(img_dir, "fragment")
    return fragment_dir


def get_data_dir(out_put_dir):
    data_dir = os.path.join(out_put_dir, "data")
    return data_dir


def get_label_data_dir(out_put_dir):
    label_data = os.path.join(out_put_dir, "label_data")
    return label_data


def get_voc_data_dir(out_put_dir):
    voc_data = os.path.join(out_put_dir, "voc_data")
    return voc_data


def get_lsvt_data_dir(out_put_dir):
    lsvt_data = os.path.join(out_put_dir, "lsvt_data")
    return lsvt_data





def gen_all_pic():
    """
    生成全部图片
    :return:
    """
    from service import conf
    gen_count = conf['base']['count_per_process']


    index = 0
    while index < gen_count:
        log.info("-" * 20 + " generate new picture {index}/{gen_count}".format(index=index+1,
                                                                               gen_count=gen_count) + "-" * 20)
        dump_data = gen_pic()

        # 写入label
        if dump_data:
            # print(dump_data)
            add_label_data(dump_data)
            # 生成voc
            if conf['base']['gen_voc']:
                gen_voc(dump_data)
                index += 1

            if conf['base']['gen_lsvt']:
                gen_lsvt(dump_data)
                index += 1
        index += 1



# 生成文本位置
def gen_pic():
    from service import layout_provider

    layout = layout_provider.gen_next_layout()

    if not layout.is_empty():
        dump_data = layout.dump()
        # layout.show(draw_rect=True)
        # print(dump_data)
        return dump_data
    else:
        log.info("-" * 10 + "layout is empty" + "-" * 10)
        return None


import csv
def add_label_data(layout_data):
    """
    写入标签文件
    :return:
    """
    from service import conf
    # 创建 output->label
    out_put_dir = conf['provider']['layout']['out_put_dir']
    # label_data_dir = get_label_data_dir(out_put_dir=out_put_dir)
    # os.makedirs(label_data_dir, exist_ok=True)

    # 创建 output->label->txt
    # label_file_path = os.path.join(label_data_dir, "label_{pid}.txt".format(pid=os.getpid()))
    # fragment_dir = get_fragment_dir(out_put_dir)

    # 拷贝图片
    # fragment_list = layout_data['fragment']

    # 写入txt
    # with open(label_file_path, 'a+') as f:
    #     for fragment in fragment_list:
    #         fragment_name = fragment['fragment_name']
    #         fragment_img_src_path = os.path.join(fragment_dir, fragment_name)
    #         fragment_img_dst_path = os.path.join(label_data_dir, fragment_name)
    #         shutil.copy(fragment_img_src_path, fragment_img_dst_path)
    #
    #         txt = fragment['data']
    #         img_name = fragment['fragment_name']
    #         line = img_name + "^" + txt + os.linesep
    #         f.write(line)


    ###########################

    # 创建my_data
    my_data = os.path.join(out_put_dir, "my_data")
    os.makedirs(my_data, exist_ok=True)

    # my_data下的文件名，创建
    pic_name = layout_data['pic_name']
    pic_name = pic_name[:-4]
    pic_dir = os.path.join(my_data, pic_name)
    os.makedirs(pic_dir, exist_ok=True)

    # 创建csv，再my_data->对应文件
    csv_file_path = os.path.join(pic_dir, f"{pic_name}_char.csv")

    # 创建csv,保存旋转后的lines的bbox
    csv_lines_file_path = os.path.join(pic_dir, f"{pic_name}_lines.csv")




    # # copy处理后图片
    # # 图像位置
    # pic_img_src_path = os.path.join(out_put_dir, 'img')
    # # 保存位置
    # shutil.copy(pic_img_src_path, pic_dir)



    # box
    fragment_list = layout_data['fragment']

    box_data = []
    string_box = []

    # 旋转后的bbox
    box_data_lines = []


    for fragment in fragment_list:
        fragment_name = fragment['fragment_name']

        # # 拷贝fragment
        # fragment_img_src_path = os.path.join(fragment_dir, fragment_name)
        # fragment_img_dst_path = os.path.join(pic_dir, fragment_name)
        # shutil.copy(fragment_img_src_path, fragment_img_dst_path)

        # 获取相关box
        box = fragment['char_boxes']
        string = fragment['data']
        # 处理后的文本bbox
        box_line = fragment['rotate_box']


        box_data.append(box)
        string_box.append(string)
        box_data_lines.append(box_line)



    with open(csv_lines_file_path, 'w', newline='') as csv_file1:
        writer = csv.writer(csv_file1)
        writer.writerow(["p1", "p2", "p3", "p4", "info"])
        # 斜放bbox
        # 左下，左上，右上，右下
        for i in range(len(box_data_lines)):
            line = box_data_lines[i]
            str_text = string_box[i]
            writer.writerow([line[1], line[2], line[3], line[0], str_text])




    # 写入char_ box 信息
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["p1", "p2", "p3", "p4", "info"])
        for i in range(len(box_data)):

            for j in range(len(box_data[i])):
                char_box = box_data[i][j]
                text = string_box[i][j]
                writer.writerow([char_box[0], char_box[1], char_box[2], char_box[3], text])






    log.info("gen label data success!")










#
# def add_label_data(layout_data):
#     """
#     写入标签文件
#     :return:
#     """
#     from service import conf
#     # 创建目录 output->label_data
#     out_put_dir = conf['provider']['layout']['out_put_dir']
#     label_data_dir = get_label_data_dir(out_put_dir=out_put_dir)
#     # print(label_data_dir)
#     os.makedirs(label_data_dir, exist_ok=True)
#
#     # label_data 下图像文件夹
#     pic_name = layout_data['pic_name']
#     pic_dir = os.path.join(label_data_dir, pic_name)
#     os.makedirs(pic_dir, exist_ok=True)
#
#     # label_data 下保存csv
#     csv_file_path = os.path.join(pic_dir, f"{pic_name}.csv")
#     # output-> img -> fragment的文件夹
#     fragment_dir = get_fragment_dir(out_put_dir)
#
#     # 拷贝图片
#     fragment_list = layout_data['fragment']
#     box_data = []
#     string_box  = []
#
#     for fragment in fragment_list:
#         fragment_name = fragment['fragment_name']
#         fragment_img_src_path = os.path.join(fragment_dir, fragment_name)
#         fragment_img_dst_path = os.path.join(pic_dir, fragment_name)
#         shutil.copy(fragment_img_src_path, fragment_img_dst_path)
#
#         #
#         box = fragment['char_boxes']
#         string = fragment['data']
#
#         box_data.append(box)
#         string_box.append(string)
#
#
#     # 写入 box 信息
#     with open(csv_file_path, 'w', newline='') as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(["p1", "p2", "p3", "p4", "info"])
#         for i in range(len(box_data)):
#             for j in range(len(box_data[i])):
#                 char_box = box_data[i][j]
#                 text = string_box[i][j]
#                 writer.writerow([char_box[0],char_box[1],char_box[2],char_box[3], text])
#         # writer.writerows(box_data)
#
#     # 使用eval
#     # with open(csv_file_path, 'r') as file:
#     #     reader = csv.DictReader(file)
#     #     for row in reader:
#     #         print(row['p1'])
#     #         p1_value = eval(row['p1'])
#     #         print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',type(p1_value), p1_value)
#     #         print(p1_value[0])
#     #         break





















def gen_voc(layout_data):
    """
    生成voc数据集
    :return:
    """
    from service import conf
    out_put_dir = conf['provider']['layout']['out_put_dir']
    voc_data_dir = get_voc_data_dir(out_put_dir=out_put_dir)

    voc_img_dir = os.path.join(voc_data_dir, "voc_img")
    voc_xml_dir = os.path.join(voc_data_dir, "voc_xml")
    os.makedirs(voc_img_dir, exist_ok=True)
    os.makedirs(voc_xml_dir, exist_ok=True)

    pic_dir = get_pic_dir(out_put_dir)
    pic_name = layout_data['pic_name']
    pic_path = os.path.join(pic_dir, pic_name)
    pic_save_to_path = os.path.join(voc_img_dir, pic_name)

    # 拷贝图片
    shutil.copy(pic_path, pic_save_to_path)
    log.info("copy img success")

    # 生成标签文本
    _gen_voc(voc_xml_dir, data=layout_data)

    log.info("voc data gen success")


def _gen_voc(save_dir, data, image_format='png'):
    w = data['width']
    h = data['height']

    node_root = Element('annotation')
    '''folder'''
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'
    '''filename'''
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = data['pic_name']
    '''source'''
    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'The VOC2007 Database'
    node_annotation = SubElement(node_source, 'annotation')
    node_annotation.text = 'PASCAL VOC2007'
    node_image = SubElement(node_source, 'image')
    node_image.text = 'flickr'
    '''size'''
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(w)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(h)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    '''segmented'''
    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'
    '''object coord and label'''
    for i, fragment in enumerate(data['fragment']):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = fragment['orientation'][0] + "_text"
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(fragment['box'][0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(fragment['box'][1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(fragment['box'][2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(fragment['box'][3])

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行

    save_xml = os.path.join(save_dir, data['pic_name'].replace(image_format, 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)


def gen_lsvt(layout_data):
    """

    :param layout_data:
    :return:
    """
    from service import conf
    out_put_dir = conf['provider']['layout']['out_put_dir']
    lsvt_data_dir = get_lsvt_data_dir(out_put_dir=out_put_dir)

    lsvt_data_img_dir = os.path.join(lsvt_data_dir, "train")
    os.makedirs(lsvt_data_img_dir, exist_ok=True)
    lsvt_json_path = os.path.join(lsvt_data_dir, "train_full_labels_{pid}.json".format(pid=os.getpid()))

    pic_dir = get_pic_dir(out_put_dir)
    pic_name = layout_data['pic_name']
    pic_path = os.path.join(pic_dir, pic_name)
    pic_save_to_path = os.path.join(lsvt_data_img_dir, pic_name)
    # 拷贝图片
    shutil.copy(pic_path, pic_save_to_path)
    log.info("copy img success")
    # 生成标签文本
    _gen_lsvt(layout_data, lsvt_json_path)
    log.info("voc data gen success")


def _gen_lsvt(layout_data, lsvt_json_path):
    """

    :param layout_data:
    :param lsvt_json_path:
    :return:
    """
    pic_name = layout_data['pic_name']
    pic_name = pic_name.split('.')[0]
    fragment_list = layout_data['fragment']
    print(lsvt_json_path)
    if not os.path.exists(lsvt_json_path):
        fp = open(lsvt_json_path, "w")
        fp.close()
    with open(lsvt_json_path, 'r') as f:
        text = f.read()
        if text == '':
            load_dict = dict()
        else:
            load_dict = json.loads(text)

    with open(lsvt_json_path, 'w') as f:
        lsvt_dict_list = list()
        for fragment in fragment_list:
            txt = fragment['data']
            rotate_box = fragment['rotate_box']
            char_boxes = fragment['char_boxes']
            lsvt_info = dict(transcription=txt, points=rotate_box, char_boxes=char_boxes, illegibility=False)
            lsvt_dict_list.append(lsvt_info)
        load_dict.update({pic_name: lsvt_dict_list})
        # f.seek(0)

        json.dump(load_dict, f)

