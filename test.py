from model.ImageNetVGG16 import ImageNetVGG16 as VGG16
from similarity import cos
from visulization import visual
import matplotlib.pyplot as plt

vgg16 = VGG16()
feature_map1 = vgg16.predict_img('/home/hack/PycharmProjects/vgg/data/paper/bim2.png', layer=4)
# feature_map2 = vgg16.extract_feature('/home/hack/PycharmProjects/vgg/data/paper/bim2.png', layer=2)
# print(feature_map.shape)

visual.vis_into_one(feature_map1)
# visual.vis_into_one(feature_map2)
photo = plt.imread("/home/hack/PycharmProjects/vgg/data/paper/bim2.png")
plt.figure(2)
plt.imshow(photo)
# visual.vis_all(feature_map)


########
feature_map1 = vgg16.predict_img('/home/hack/PycharmProjects/vgg/data/paper/photo2.png', layer=4)
# feature_map2 = vgg16.extract_feature('/home/hack/PycharmProjects/vgg/data/paper/bim2.png', layer=2)
# print(feature_map.shape)
visual.vis_into_one(feature_map1)
# visual.vis_into_one(feature_map2)
photo2 = plt.imread("/home/hack/PycharmProjects/vgg/data/paper/photo2.png")
plt.figure(4)
plt.imshow(photo2)
# visual.vis_all(feature_map)
plt.show()
