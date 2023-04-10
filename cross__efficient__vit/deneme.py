import detect_faces
import os


def deneme():
    detect_faces.main(data_path="cross__efficient__vit/run_test")
    os.system('python preprocessing/extract_crops.py --data_path cross__efficient__vit/run_test --output_path cross__efficient__vit/run_test')