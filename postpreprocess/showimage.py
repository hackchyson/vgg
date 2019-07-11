import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
import logging
from model.ImageNetVGG16 import ImageNetVGG16 as VGG16
from similarity import cos
from visulization import visual
import matplotlib.pyplot as plt
from utils.mix import mix_feature
import pickle
import preprocess.data as data
from similarity.cos import max_sim

logging.basicConfig(level=logging.INFO)


class Picture(QWidget):
    def __init__(self, bims_path, labels_path, filenames_path):
        super(Picture, self).__init__()

        self.resize(1200, 600)
        self.setWindowTitle("LOCALIZATION")

        self.label = QLabel(self)
        self.label.setText("AREA TO SHOW ORIGINAL PHOTO")
        self.label.setFixedSize(400, 400)
        self.label.move(100, 100)
        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(100,100,100);font-size:14px;font-weight:bold;}")

        self.label3 = QLabel(self)
        self.label3.setText("AREA TO SHOW BIM MODEL CAPTURES")
        self.label3.setFixedSize(400, 400)
        self.label3.move(700, 100)
        self.label3.setStyleSheet("QLabel{background:white;}"
                                  "QLabel{color:rgb(100,100,100);font-size:14px;font-weight:bold;}")

        self.label2 = QLabel(self)
        self.label2.setText("AREA TO SHOW LOCALIZATION")
        self.label2.setFixedSize(400, 30)
        self.label2.move(700, 530)
        self.label2.setStyleSheet("QLabel{background:white;}"
                                  "QLabel{color:rgb(100,100,100);font-size:14px;font-weight:bold;}")
        self.center()
        btn = QPushButton(self)
        btn.setText("OPEN A IMAGE")
        btn.move(30, 30)
        btn.clicked.connect(self.open_image)

        # deep learning
        self.vgg16 = VGG16()
        self.features = self.vgg16.extract_features(bims_path)
        self.labels = data.load(labels_path)
        self.filenames = data.load(filenames_path)

    def open_image(self):
        img_name, img_type = QFileDialog.getOpenFileName(self, "OPEN A IMAGE", "/home/hack/Pictures",
                                                         "*.jpg; *.png;;All Files(*)")
        jpg = QtGui.QPixmap(img_name).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)

        goal_feature_map = self.vgg16.extract_feature(img_name)
        goal_feature_map = mix_feature(goal_feature_map)

        sim_idx, _ = max_sim(goal_feature_map, self.features)
        sim_loc = self.labels[sim_idx]
        sim_pic = self.filenames[sim_idx]
        # jpg = QtGui.QPixmap(img_name).scaled(self.label.width(), self.label.height())
        jpg3 = QtGui.QPixmap(sim_pic).scaled(self.label3.width(), self.label3.height())

        logging.info("image name: {}".format(img_name))
        logging.info("image type: {}".format(img_type))
        self.label3.setPixmap(jpg3)
        self.label2.setText(sim_loc)

    def center(self):
        """
        Locate the widget center of the screen.
        :return:
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = Picture('/home/hack/PycharmProjects/vgg/data/bim/paper/data.npy',
                 '/home/hack/PycharmProjects/vgg/data/bim/paper/label.pickle',
                 '/home/hack/PycharmProjects/vgg/data/bim/paper/bims.pickle')
    my.show()
    sys.exit(app.exec_())
