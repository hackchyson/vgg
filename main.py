from model.ImageNetVGG16 import ImageNetVGG16 as VGG16
from similarity import cos
from visulization import visual

vgg16 = VGG16(layer=2)
feature_map = vgg16.predict_img('data/dog.jpg')
print(feature_map.shape)

visual.vis_into_one(feature_map)
visual.vis_all(feature_map)