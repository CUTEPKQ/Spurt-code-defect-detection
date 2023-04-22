from ruamel import yaml

from service.provider.TextImgProvider import TextImgProvider
from service.provider.BackgroundImgProvider import BackgroundImgProvider
from service.provider.TextProvider import TextProvider
from service.provider.SmoothAreaProvider import SmoothAreaProvider
from service.provider.LayoutProvider import LayoutProvider
from utils import log
from multiprocessing import Pool
import traceback
import os

basedir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
conf: dict
text_img_provider: TextImgProvider
background_img_provider: BackgroundImgProvider
text_provider: TextProvider
smooth_area_provider: SmoothAreaProvider
layout_provider: LayoutProvider


def load_from_config():
    global text_img_provider
    global background_img_provider
    global text_provider
    global smooth_area_provider
    global layout_provider

    # 文字图像(这个char绘制)
    text_img_provider = TextImgProvider(**conf['provider']['text_img'])

    # 背景
    background_img_provider = BackgroundImgProvider(conf['provider']['bg_img'])

    # 文本
    text_provider = TextProvider(conf['provider']['text'])

    # 特征点，光滑区域，用来绘制图像
    smooth_area_provider = SmoothAreaProvider(**conf['provider']['smooth_area'])

    #
    layout_provider = LayoutProvider(**conf['provider']['layout'])

# 初始化
def init_config():
    log.info("load config")
    global conf
    with open(os.path.join(basedir, "config.yml"), 'r') as f:
        conf = yaml.load(f.read(), Loader=yaml.Loader)
        load_from_config()


def init():
    init_config()


def run():
    try:
        from service.base import gen_all_pic
        gen_all_pic()
    except Exception as e:
        traceback.print_exc()





import string
# 英文字典
# 数字，英文大小写
en_dic = string.printable
# print(en_dic)

en_dic_list = list(en_dic)
en_dic_list = en_dic_list[0:62]


def create_text(line_num = 5):
    """
    # 个数数随机(1,17)
     字符随机
    :return: text
    """
    import random


    for item in conf["provider"]["text"]:
        if item["name"] == "english":
            text_path = item["path"]

            with open(text_path, 'w') as f:
                # 对应路径
                for i in range(line_num):
                    num = random.randint(1, 17)
                    text = ''
                    # 创建需要的字符串
                    for i in range(num):
                        random_char = random.choice(en_dic_list)
                        text += random_char

                    f.write(text + '\n')

                    # print(text_path)

            break




def start():
    # 初始化，个from config
    init()

    # 线程数目
    process_count = conf['base']['process_count']
    print('Parent process {pid}.'.format(pid=os.getpid()))
    print('process count : {process_count}'.format(process_count=process_count))

    # 创建随机的字符串个数，默认5
    create_text(14)


    # 线程池
    p = Pool(process_count)

    # 每个线程执行run.
    for i in range(process_count):
        p.apply_async(run)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


if __name__ == '__main__':
    init_config()
    print(conf)
