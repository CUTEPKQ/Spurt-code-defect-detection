import paddlex as pdx
from paddlex import transforms as T

model = pdx.load_model('output/mobilenetv3_small/best_model')
test_transforms = T.Compose([
    T.ResizeByShort(short_size=256), T.CenterCrop(crop_size=224), T.Normalize()
])

test_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/test_list.txt',
    label_list='vegetables_cls/labels.txt',
    )
# model.test(test_dataset)

# for file in test_dataset[]:

result = model.predict(test_dataset)
# result = model.predict('vegetables_cls/bocai/100.jpg')
print("Predict Result: ", result)