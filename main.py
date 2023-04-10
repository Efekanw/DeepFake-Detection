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

def runn():
    detect_faces.main(data_path="run_test")
    os.system('python preprocessing/extract_crops.py --data_path "cross__efficient__vit/run_test" --output_path "cross__efficient__vit/run_test/run_folder"')
    os.system("python cross__efficient__vit/run.py --model_path cross__efficient__vit/cross_efficient_vit.pth --config cross__efficient__vit/configs/architecture.yaml")


#detect_faces.main(data_path="dataset/test_set")
#os.system('python preprocessing/extract_crops.py --data_path "dataset/test_set" --output_path "dataset/test_set/DFDC"')


#os.system("python cross__efficient__vit/test.py --model_path cross__efficient__vit/cross_efficient_vit.pth --config cross__efficient__vit/configs/architecture.yaml")

#os.system("python cross__efficient__vit/run.py --model_path cross__efficient__vit/cross_efficient_vit.pth --config cross__efficient__vit/configs/architecture.yaml")

#runn()
start()