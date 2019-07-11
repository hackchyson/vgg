import sys
from PyQt5.QtWidgets import QApplication, QWidget

if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = QWidget()
    w.resize(500, 550)
    w.move(300, 300)
    w.setWindowTitle('LOCALIZATION')
    w.show()

    # The exec_() method has an underscore.
    # It is because the exec is a Python keyword. And thus, exec_() was used instead.
    sys.exit(app.exec_())
