import os
import re
import torch
from CNN.kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video
from CNN.training.zoo.classifiers import DeepFakeClassifier


def inference(video_path):
    weights_dir = "CNN/weights"
    models = ["final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36"]
              #"final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
              # "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29",
              # "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31",
              # "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37",
              # "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40",
              # "final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23"]
    model_paths = [os.path.join(weights_dir, model) for model in models]
    models.clear()
    for path in model_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        model.eval()
        del checkpoint
        models.append(model)

    frames_per_video = 32
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)
    input_size = 380
    strategy = confident_strategy
    video_name = video_path.rsplit('/', 1)[-1]
    predictions = predict_on_video(face_extractor=face_extractor, video_path=video_path, input_size=input_size,
                                    batch_size=frames_per_video, models=models, strategy=strategy, apply_compression= False)
    results = {"filename": video_name, "label": predictions}
    return results
