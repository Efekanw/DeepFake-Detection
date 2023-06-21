import argparse
import os
from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool
from os import cpu_count

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations import Compose, PadIfNeeded
from progress.bar import Bar
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from tqdm import tqdm

from cross__efficient__vit.cross_efficient_vit import CrossEfficientViT
from cross__efficient__vit.transforms.albu import IsotropicResize
from cross__efficient__vit.utils import custom_round, custom_video_round
from database import db_functions, db_connection
#########################
####### CONSTANTS #######
#########################

MODELS_DIR = "models"
BASE_DIR = "../"
DATA_DIR = "run"
TEST_DIR = DATA_DIR + '/' + "test_crops"
OUTPUT_DIR = "models" + '/' + "tests"
#OUTPUT_DIR = os.path.join(MODELS_DIR, "tests")
TEST_LABELS_PATH = DATA_DIR + '/' + "dfdc_test_labels.txt"
#TEST_LABELS_PATH = os.path.join(BASE_DIR, "dataset/dfdc_test_labels.csv")

#########################
####### UTILITIES #######
#########################
connection = db_connection.connect("deepfakedetection", "postgres", "123")

def save_confusion_matrix(confusion_matrix, model_name):
  try:
      matplotlib.use('agg')
      fig, ax = plt.subplots()
      im = ax.imshow(confusion_matrix, cmap="Blues")

      threshold = im.norm(confusion_matrix.max())/2.
      textcolors=("black", "white")

      ax.set_xticks(np.arange(2))
      ax.set_yticks(np.arange(2))
      ax.set_xticklabels(["original", "fake"])
      ax.set_yticklabels(["original", "fake"])

      ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

      for i in range(2):
          for j in range(2):
              text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center",
                             fontsize=12, color=textcolors[int(im.norm(confusion_matrix[i, j]) > threshold)])

      fig.tight_layout()
      # global OUTPUT_DIR
      # folder_name = model_name.split('.')[0]
      # OUTPUT_DIR += '/' + folder_name
      image_path = os.path.join(OUTPUT_DIR, "confusion.jpg")
      plt.savefig(image_path)
      return image_path
  except Exception as e:
      print(e)
  

def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
  try:
      matplotlib.use('agg')
      plt.figure(1)
      plt.plot([0, 1], [0, 1], 'k--')

      fpr, tpr, th = metrics.roc_curve(correct_labels, preds)

      model_auc = auc(fpr, tpr)


      plt.plot(fpr, tpr, label="Model_"+ model_name + ' (area = {:.3f})'.format(model_auc))

      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve')
      plt.legend(loc='best')
      # global OUTPUT_DIR
      folder_name = model_name.split('.')[0]
      # OUTPUT_DIR += '/' + folder_name
      roc_path = os.path.join(OUTPUT_DIR, "roc.jpg")
      plt.savefig(roc_path)
      plt.clf()
      return roc_path
  except Exception as e:
      print(e)


def read_frames(video_path, videos, frames_per_video, config, run):
    
    # Get the video label based on dataset selected
    #method = get_method(video_path, DATA_DIR)
    #if "Original" in video_path:
    #    label = 0.
    if run == 0:
        test_df = pd.read_csv(TEST_LABELS_PATH, sep=",")
        video_folder_name = os.path.basename(video_path)
        video_key = video_folder_name + ".mp4"
        #label = test_df.loc[test_df['filename'] == video_key]['label'].values[0]
        print(video_key)
        label = test_df.loc[test_df['filename'] == video_key, 'label'].iloc[0]
    else:
        label = 1.
    

    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    frames_interval = int(frames_number / frames_per_video)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video

    for path in frames_paths:
        for i in range(0,3): # Consider up to 3 faces per video
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)

    # Select only the frames at a certain interval
    if frames_interval > 0:
        for key in frames_paths_dict.keys():
            if len(frames_paths_dict) > frames_interval:
                frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
            
            frames_paths_dict[key] = frames_paths_dict[key][:frames_per_video]

    # Select N frames from the collected ones
    video = {}
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            transform = create_base_transform(config['model']['image-size'])
            image = transform(image=cv2.imread(os.path.join(video_path, frame_image)))['image']
            if len(image) > 0:
                if key in video:
                    video[key].append(image)
                else:
                    video[key] = [image]
    videos.append((video, label, video_path))


def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

#########################
#######   MODEL   #######
#########################


# Main body
def main(config, batch_size, model_name, userid, model_path):
    
    # parser = argparse.ArgumentParser()
    #
    # # parser.add_argument('--workers', default=0, type=int,
    # #                     help='Number of data loader workers.')
    # parser.add_argument('--model_path', default='', type=str, metavar='PATH',
    #                     help='Path to model checkpoint (default: none).')
    # # parser.add_argument('--dataset', type=str, default='DFDC',
    # #                     help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|DFDC)")
    # # parser.add_argument('--max_videos', type=int, default=-1,
    # #                     help="Maximum number of videos to use for training (default: all).")
    # # parser.add_argument('--config', type=str,
    # #                     help="Which configuration to use. See into 'config' folder.")
    # # parser.add_argument('--efficient_net', type=int, default=0,
    # #                     help="Which EfficientNet version to use (0 or 7, default: 0)")
    # parser.add_argument('--frames_per_video', type=int, default=30,
    #                     help="How many equidistant frames for each video (default: 30)")
    # parser.add_argument('--batch_size', type=int, default=32,
    #                     help="Batch size (default: 32)")
    #
    # opt = parser.parse_args()
    # print(opt)
    
    with open(config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
 
        
    if os.path.exists(model_path):
        model = CrossEfficientViT(config=config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        #model = model.cuda()
    else:
        print("No model found.")
        exit()

    #model_name = os.path.basename(opt.model_path)
    model_name = model_name
    #DATA_DIR = data_path

    #########################
    ####### EXECUTION #######
    #########################


    #OUTPUT_DIR = os.path.join(OUTPUT_DIR, opt.dataset)
    
    # if not os.path.exists(OUTPUT_DIR):
    #     os.makedirs(OUTPUT_DIR)
   


    NUM_CLASSES = 1
    preds = []

    paths = []
    videos = []


    # Read all videos paths
    for index, video_folder in enumerate(os.listdir(TEST_DIR)):
        paths.append(os.path.join(TEST_DIR, video_folder))

    # Read faces
    # with Pool(processes=cpu_count()-1) as p:
    #     with tqdm(total=len(paths)) as pbar:
    #         for v in p.imap_unordered(partial(read_frames, videos=videos, config=config, run=0),paths):
    #             pbar.update()

    for path in paths:
        read_frames(video_path=path, videos=videos, config=config, run=0, frames_per_video=30)

    video_names = np.asarray([row[2] for row in videos])
    correct_test_labels = np.asarray([row[1] for row in videos])
    videos = np.asarray([row[0] for row in videos])
    preds = []
    global OUTPUT_DIR
    OUTPUT_DIR = "models" + '/' + "tests"
    folder_name = model_name.split('.')[0]
    OUTPUT_DIR += '/' + folder_name
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    bar = Bar('Predicting', max=len(videos))

    f = open(OUTPUT_DIR+ "/" + "_" + model_name + "_labels.txt", "w+")
    for index, video in enumerate(videos):
        video_faces_preds = []
        video_name = video_names[index]
        f.write(video_name)
        for key in video:
            faces_preds = []
            video_faces = video[key]
            for i in range(0, len(video_faces), batch_size):
                faces = video_faces[i:i+batch_size]
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
            del faces
            del scaled_pred
            current_faces_pred = sum(faces_preds)/len(faces_preds)
            face_pred = current_faces_pred.cpu().detach().numpy()[0]
            f.write(" " + str(face_pred))
            video_faces_preds.append(face_pred)
        del faces_preds
        del current_faces_pred
        bar.next()
        if len(video_faces_preds) > 1:
            video_pred = custom_video_round(video_faces_preds)
        else:
            video_pred = video_faces_preds[0]
        preds.append([video_pred])

        f.write(" --> " + str(video_pred) + "(CORRECT: " + str(correct_test_labels[index]) + ")" +"\n")
        del video_faces_preds
    f.close()
    bar.finish()


    #########################
    #######  METRICS  #######
    #########################

    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensor_labels = torch.tensor([[float(label)] for label in correct_test_labels])
    tensor_preds = torch.tensor(preds)


    loss = loss_fn(tensor_preds, tensor_labels).numpy()
    loss = loss.item()
    loss = round(loss % 1, 2)
    #accuracy = accuracy_score(np.asarray(preds).round(), correct_test_labels) # Classic way
    accuracy = accuracy_score(custom_round(np.asarray(preds)), correct_test_labels) # Custom way
    f1 = f1_score(correct_test_labels, custom_round(np.asarray(preds)))
    #accuracy = 1
    #loss = 1
    #model_name = 'deneme'
    #f1 = 1
    print(model_name, "Test Accuracy:", accuracy, "Loss:", loss, "F1", f1)
    roc_path = save_roc_curves(correct_test_labels, preds, model_name, accuracy, loss, f1)
    conf_path = save_confusion_matrix(metrics.confusion_matrix(correct_test_labels,custom_round(np.asarray(preds))), model_name=model_name)
    #conf_path = os.path.join(OUTPUT_DIR, "confusion.jpg")
    #roc_path = os.path.join(OUTPUT_DIR, "roc.jpg")
    result_dict = {"metric_name": model_name,"accuracy": accuracy, "loss": loss, "f1": f1}
    db_functions.insert_metrics(connection=connection, conf_image_path=conf_path, roc_image_path=roc_path, userid=userid, accuracy=accuracy, f1=f1, loss=loss, metric_name=model_name, model_path=model_path)
    return result_dict
