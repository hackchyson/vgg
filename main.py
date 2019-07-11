from model.ImageNetVGG16 import ImageNetVGG16 as VGG16
from similarity import cos
from visulization import visual
import matplotlib.pyplot as plt

vgg16 = VGG16()
feature_map1 = vgg16.predict_img('data/dog.jpg', layer=3)
feature_map2 = vgg16.extract_feature('data/dog.jpg', layer=3)
# print(feature_map.shape)

visual.vis_into_one(feature_map1)
visual.vis_into_one(feature_map2)
# visual.vis_all(feature_map)
plt.show()
