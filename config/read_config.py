'''
Author: fengyx ,fengyx@hnu.edu.cn
Date: 2023-04-24 20:40:12
LastEditors: fengyx ,fengyx@hnu.edu.cn
LastEditTime: 2023-04-24 20:40:32
Description: 
'''
import yaml
import os


'''
description: this function read the config and return a dict including the paramaters
param {*} config_name
return {*} dict
'''
def read_config(config_name) -> dict:
    name = config_name + '.yaml'
    curPath = os.path.dirname(os.path.realpath(__file__))
    yamlPath = os.path.join(curPath, name)
    f = open(yamlPath, 'r', encoding='utf-8')
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config