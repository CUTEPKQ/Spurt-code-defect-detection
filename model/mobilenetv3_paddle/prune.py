import paddlex as pdx
from paddlex import transforms as T

# 下载和解压蔬菜分类数据集
veg_dataset = 'https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz'
pdx.utils.download_and_decompress(veg_dataset, path='./')

# 定义训练和验证时的transforms
# API说明https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
train_transforms = T.Compose(
    [T.RandomCrop(crop_size=224), T.RandomHorizontalFlip(), T.Normalize()])

eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256), T.CenterCrop(crop_size=224), T.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
train_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/train_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/val_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=eval_transforms)

# 加载模型
model = pdx.load_model('output/mobilenetv3_small/best_model')

# Step 1/3: 分析模型各层参数在不同的剪裁比例下的敏感度
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/classification.md#analyze_sensitivity
model.analyze_sensitivity(
    dataset=eval_dataset, save_dir='output/mobilenetv3_small/prune')

# Step 2/3: 根据选择的FLOPs减小比例对模型进行剪裁
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/classification.md#prune
model.prune(pruned_flops=.2)

# Step 3/3: 对剪裁后的模型重新训练
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/classification.md#train
# 各参数介绍与调整说明：https://github.com/PaddlePaddle/PaddleX/tree/develop/docs/parameters.md
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    lr_decay_epochs=[4, 6, 8],
    learning_rate=0.01,
    pretrain_weights=None,
    save_dir='output/mobilenetv3_small/prune',
    use_vdl=True)