import os
import detect_faces
import sys
from PyQt5.QtWidgets import QApplication
from DeepFake_UI_connection import LoginWindow, RegisterWindow
from database import db_connection

connection = db_connection.connect("deepfakedetection", "postgres", "123")


def start():
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    w = LoginWindow(connection)
    w.show()
    sys.exit(app.exec_())

start()