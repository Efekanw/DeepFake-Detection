import extract_crops
import detect_faces
from cross__efficient__vit import run


def visionTransformerPredict(file_path):
    detect_faces.main(data_path="run_test", video_path=file_path)
    extract_crops.main(data_path="run_test", output_path="run_test/run_folder", dataset="DFDC", file_path=file_path)
    a = run.main(model_path="cross__efficient__vit/cross_efficient_vit.pth", config="cross__efficient__vit/configs/architecture.yaml",file_path=file_path)
    return str(a)