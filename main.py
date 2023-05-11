
import paddlex as pdx
import cv2

from opt import get_opts
from paddlex import transforms as T


class MyModel:
    def __init__(self,hparams) -> None:
        self.hparams = hparams
        self.model = pdx.cls.MobileNetV3_small(num_classes=self.hparams.num_classes)

    def train(self,
              train_dataset,
              eval_dataset,
              lr_decay_epochs) -> None:

        self.model.train(num_epochs = self.hparams.num_epochs,                       
                          train_dataset = train_dataset,
                          train_batch_size = self.hparams.train_batch_size,
                          eval_dataset = eval_dataset,
                          lr_decay_epochs = lr_decay_epochs,
                          learning_rate = 0.01,
                          save_dir = self.hparams.save_path,
                          use_vdl = self.hparams.ues_vld)

    def load_model(self,path) -> None:
        self.model = pdx.load_model(path)
    
    def predict(self,img) :
        return self.model.predict(img)


if __name__=='__main__':
    hparams = get_opts()

    train_transforms = T.Compose(
    [T.RandomCrop(crop_size=224), T.RandomHorizontalFlip(), T.Normalize()])
    eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256), T.CenterCrop(crop_size=224), T.Normalize()])

    train_dataset = pdx.datasets.ImageNet(
    data_dir='data',
    file_list='data/train_list.txt',
    label_list='data/labels.txt',
    transforms=train_transforms,
    shuffle=True)
    hparams.num_classes = len(train_dataset.labels)

    eval_dataset = pdx.datasets.ImageNet(
    data_dir='data',
    file_list='data/val_list.txt',
    label_list='data/labels.txt',
    transforms=eval_transforms)

    mymodel = MyModel(hparams)
    mymodel.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8],
            use_vdl=True)

    