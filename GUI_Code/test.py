from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton

class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Button Click")
        layout = QVBoxLayout()
        self.button = QPushButton("Click Me")
        self.button.clicked.connect(self.on_click)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def on_click(self):
        print("Button clicked!")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = TestWidget()
    win.show()
    sys.exit(app.exec_())
