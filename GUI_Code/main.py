# main.py

import sys
import os
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from widgets.single_image_widget import SingleImageWidget
from widgets.sweep_process_widget import SweepProcessWidget
from models.seg_model import model as build_seg_model
from models.error_regression_model import build_residual_model

def resource_path(relative_path):
    """获取资源的绝对路径，兼容 PyInstaller """
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fetal AC Estimation System")
        self.setFixedSize(1000, 600)

        # ✅ 只加载一次模型并共享
        self.seg_model = build_seg_model(img_height=384, img_width=508, classes=2)
        self.seg_model.load_weights(resource_path("assets/seg_model_weights.h5"))

        self.residual_model = build_residual_model()
        self.residual_model.load_weights(resource_path("assets/error_regression_model.h5"))

        # ✅ 传入共享模型
        self.single_widget = SingleImageWidget(self.seg_model, self.residual_model)
        self.sweep_widget = SweepProcessWidget(self.seg_model, self.residual_model)

        tabs = QTabWidget()
        tabs.addTab(self.single_widget, "Single Frame Analysis")
        tabs.addTab(self.sweep_widget, "Sweep Process")

        self.setCentralWidget(tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
