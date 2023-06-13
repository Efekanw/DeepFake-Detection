import os
import yaml
import torch
from cross__efficient__vit.cross_efficient_vit import CrossEfficientViT
import numpy as np
from cross__efficient__vit.utils import custom_video_round
from cross__efficient__vit.test import read_frames

RUN_DIR = "run_test/run_folder"
batch_size = 32


def main(model_path, config, file_path, batch_size=32):

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', default='', type=str, metavar='PATH',
    #                     help='Path to model checkpoint (default: none).')
    # parser.add_argument('--config', type=str,
    #                     help="Which configuration to use. See into 'config' folder.")
    # parser.add_argument('--batch_size', type=int, default=32,
    #                     help="Batch size (default: 32)")
    # opt = parser.parse_args()


    with open(config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    if os.path.exists(model_path):
        model = CrossEfficientViT(config=config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        # model = model.cuda()
    else:
        print("No model found.")
        exit()

    model_name = os.path.basename(model_path)

    #mgr = Manager()
    #videos = mgr.list()
    videos = []
    paths = []
    # if file_path != '':
    #     path = file_path
    #     read_frames(video_path=path, videos=videos, frames_per_video=30, config=config, run=1)
    # else:
    method_folder = os.path.join(RUN_DIR)
    print(method_folder)
    for index, video_folder in enumerate(os.listdir(method_folder)):
        paths.append(os.path.join(method_folder, video_folder))

    # Read faces
    # with Pool(processes=cpu_count()-1) as p:
    #     with tqdm(total=len(paths)) as pbar:
    #         for v in p.imap_unordered(partial(read_frames, videos=videos, frames_per_video=30 , config=config, run=1),paths):
    #             pbar.update()
    for path in paths:
        read_frames(video_path=path, videos=videos, frames_per_video=30, config=config, run=1)


    print(videos)
    video_name = [row[2] for row in videos][0]
    correct_test_label = [row[1] for row in videos][0]
    video = [row[0] for row in videos][0]
    # print(video)
    print(video_name)
    print(correct_test_label)
    #f.write(video_name)
    video_faces_preds = []
    for key in video:
        # print(key)
        faces_preds = []
        video_faces = video[key]
        for i in range(0, len(video_faces),batch_size):
            faces = video_faces[i:i + batch_size]
            faces = torch.tensor(np.asarray(faces))
            if faces.shape[0] == 0:
                continue
            faces = np.transpose(faces, (0, 3, 1, 2))
            faces = faces.float()

            pred = model(faces)

            scaled_pred = []
            for idx, p in enumerate(pred):
                scaled_pred.append(torch.sigmoid(p))
            faces_preds.extend(scaled_pred)

        current_faces_pred = sum(faces_preds) / len(faces_preds)
        face_pred = current_faces_pred.cpu().detach().numpy()[0]
        print(face_pred)
        video_faces_preds.append(face_pred)
    if len(video_faces_preds) > 1:
        video_pred = custom_video_round(video_faces_preds)
    else:
        video_pred = video_faces_preds[0]
    print("RESULT PREDICTION---")
    print(video_pred)
    return video_pred
    # if float(video_pred) > 0.5:
    #     return "Fake"
    # else:
    #     return "Real"