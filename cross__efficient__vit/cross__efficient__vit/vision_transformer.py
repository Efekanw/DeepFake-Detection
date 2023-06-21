from cross__efficient__vit.preprocessing import extract_crops
from cross__efficient__vit.preprocessing import detect_faces
from  cross__efficient__vit import test
import os
from cross__efficient__vit import run, train
import numpy as np

def visionTransformerPredict(file_path):
    detect_faces.main(data_path="run_test", video_path=file_path)
    extract_crops.main(data_path="run_test", output_path="run_test/run_folder", dataset="DFDC", file_path=file_path)
    result_predict = run.main(model_path="cross_efficient_vit.pth", config="cross__efficient__vit/configs/architecture.yaml",file_path=file_path)
    return str(result_predict)


def visionTransformerTrain(num_epochs, train_path, val_path, test_path, training_params, model_name, userid, patience=5, resume="cross__efficient__vit/cross_efficient_vit.pth"):
    # for dosya in os.listdir(train_path):
    #     dosya_yolu = train_path + '/' + dosya
    #     if os.path.isfile(dosya_yolu):
    #         print("pass")
    #         detect_faces.main(data_path='run', video_path=dosya_yolu)
    #         extract_crops.main(data_path="run", output_path="run/train_crops", dataset="DFDC", file_path=dosya_yolu)
    # for dosya in os.listdir(val_path):
    #     dosya_yolu = val_path + '/' + dosya
    #     if os.path.isfile(dosya_yolu):
    #         print("pass")
    #         detect_faces.main(data_path='run', video_path=dosya_yolu)
    #         extract_crops.main(data_path="run", output_path="run/val_crops", dataset="DFDC", file_path=dosya_yolu)
    # for dosya in os.listdir(test_path):
    #     dosya_yolu = test_path + '/' + dosya
    #     if os.path.isfile(dosya_yolu):
    #         print("pass")
    #         detect_faces.main(data_path='run', video_path=dosya_yolu)
    #         extract_crops.main(data_path="run", output_path="run/test_crops", dataset="DFDC", file_path=dosya_yolu)
    print(np.__version__)
    model_path = train.main(num_epochs=num_epochs, config="cross__efficient__vit/configs/architecture.yaml", training_params=training_params, model_name=model_name, patience=5, resume="cross_efficient_vit.pth")
    #model_path = r"C:\Users\Efekan\Documents\GitHub\DeepFake-Detection\models\efficientnet_checkpoint0_deneme"
    print(model_path)
    result_test = test.main(model_name=model_name, model_path=model_path, batch_size=training_params["bs"], config="cross__efficient__vit/configs/architecture.yaml", userid=userid)
    return result_test