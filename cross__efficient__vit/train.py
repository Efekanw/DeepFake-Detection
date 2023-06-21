import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, choice
import numpy as np
import os
import json
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from cross__efficient__vit.cross_efficient_vit import CrossEfficientViT
import uuid
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
import cv2
import glob
import pandas as pd
from tqdm import tqdm
from cross__efficient__vit.utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
from sklearn.utils.class_weight import compute_class_weight 
from torch.optim import lr_scheduler
import collections
from cross__efficient__vit.deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse
import re

BASE_DIR = '../'
DATA_DIR = "run"
#DATA_DIR = os.path.join(BASE_DIR, "user_dataset")
TRAINING_DIR = DATA_DIR + '/' + 'train_crops'
#TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = DATA_DIR + '/' + 'val_crops'
#VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
MODELS_PATH = "models"
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata") # Folder containing all training metadata for DFDC dataset
TEST_DIR = DATA_DIR + '/' + "test_set"
TRAINING_LABELS_PATH = DATA_DIR + '/' + "dfdc_train_labels.txt"
VALIDATION_LABELS_PATH = DATA_DIR + '/' + "dfdc_val_labels.txt"


def read_frames(video_path, train_dataset, validation_dataset, config, training_dir, training_labels_path, validation_dir, val_labels_path):
    
    # Get the video label based on dataset selected
    # method = get_method(video_path, DATA_DIR)
    if training_dir in video_path:
        train_df = pd.read_csv(training_labels_path, sep=",")
        print(train_df.head())

        video_folder_name = os.path.basename(video_path)
        video_folder_name += '.mp4'
        label = train_df.loc[train_df['filename'] == video_folder_name]['label'].values[0]

        print(label)
        # if "Original" in video_path:
        #     label = 0.
        # elif "DFDC" in video_path:
        #     for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
        #         with open(json_path, "r") as f:
        #             metadata = json.load(f)
        #         video_folder_name = os.path.basename(video_path)
        #         video_key = video_folder_name + ".mp4"
        #         if video_key in metadata.keys():
        #             item = metadata[video_key]
        #             label = item.get("label", None)
        #             if label == "FAKE":
        #                 label = 1.
        #             else:
        #                 label = 0.
        #             break
        #         else:
        #             label = None
    else:
        val_df = pd.read_csv(val_labels_path, sep=",")
        video_folder_name = os.path.basename(video_path)
        video_folder_name += '.mp4'
        label = val_df.loc[val_df['filename'] == video_folder_name]['label'].values[0]

    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    if label == 0:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-real']),1) # Compensate unbalancing
    else:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-fake']),1)

    
    
    if validation_dir in video_path:
        min_video_frames = int(max(min_video_frames/8, 2))
    frames_interval = int(frames_number / min_video_frames)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0,1):
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
            
            frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]
    # Select N frames from the collected ones
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            #image = transform(np.asarray(cv2.imread(os.path.join(video_path, frame_image))))
            image = cv2.imread(os.path.join(video_path, frame_image))
            if image is not None:
                if TRAINING_DIR in video_path:
                    train_dataset.append((image, label))
                else:
                    validation_dataset.append((image, label))

# Main body
def main(num_epochs, config, training_params, model_name, patience=5, resume="cross_efficient_vit.pth"):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--num_epochs', default=300, type=int,
    #                     help='Number of training epochs.')
    # # parser.add_argument('--workers', default=10, type=int,
    # #                     help='Number of data loader workers.')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='Path to latest checkpoint (default: none).')
    # # parser.add_argument('--dataset', type=str, default='All',
    # #                     help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All)")
    # # parser.add_argument('--max_videos', type=int, default=-1,
    # #                     help="Maximum number of videos to use for training (default: all).")
    # parser.add_argument('--config', type=str,
    #                     help="Which configuration to use. See into 'config' folder.")
    # # parser.add_argument('--efficient_net', type=int, default=0,
    # #                     help="Which EfficientNet version to use (0 or 7, default: 0)")
    # parser.add_argument('--patience', type=int, default=5,
    #                     help="How many epochs wait before stopping for validation loss not improving.")
    #
    # opt = parser.parse_args()
    # print(opt)

    with open(config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # global DATA_DIR
    #
    # TRAINING_DIR = DATA_DIR + '/' + "training_set"
    # VALIDATION_DIR = DATA_DIR + '/' + "validation_set"
    # TEST_DIR = DATA_DIR + '/' + "test_set"
    # TRAINING_LABELS_PATH = DATA_DIR + '/' + "dfdc_train_labels.txt"
    # VALIDATION_LABELS_PATH = DATA_DIR + '/' + "dfdc_val_labels.txt"

    model = CrossEfficientViT(config=config)
    model.train()   
    
    optimizer = torch.optim.SGD(model.parameters(), lr=training_params['lr'], weight_decay=training_params['weight_decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=training_params['step_size'], gamma=training_params['gamma'])
    starting_epoch = 0
    print(os.getcwd())
    if os.path.exists(resume):
        try:
            model.load_state_dict(torch.load(resume, map_location=torch.device('cpu')))
            starting_epoch = 0
        except Exception as e:
            print(e)
    else:
        print("No checkpoint loaded.")


    print("Model Parameters:", get_n_params(model))
   
    # #READ DATASET
    # if opt.dataset != "All":
    #     folders = ["Original", opt.dataset]
    # else:
    #     folders = ["Original", "DFDC", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

    sets = [TRAINING_DIR, VALIDATION_DIR]

    paths = []
    for dataset in sets:
        try:
            for index, video_folder_name in enumerate(os.listdir(dataset)):
                # if index == max_videos:
                #     break
                path = dataset + '/' + video_folder_name
                paths.append(path)
        except Exception as e:
            print(e)
                

    #mgr = Manager()
    # train_dataset = mgr.list()
    # validation_dataset = mgr.list()

    train_dataset = []
    validation_dataset = []

    # with Pool(processes=10) as p:
    #     with tqdm(total=len(paths)) as pbar:
    #         for v in p.imap_unordered(partial(read_frames, train_dataset=train_dataset, validation_dataset=validation_dataset),paths):
    #             pbar.update()

    for path in paths:
        read_frames(video_path=path, train_dataset=train_dataset, validation_dataset=validation_dataset,
                    config=config, training_dir=TRAINING_DIR, training_labels_path=TRAINING_LABELS_PATH,
                    validation_dir=VALIDATION_DIR, val_labels_path=VALIDATION_LABELS_PATH)

    train_samples = len(train_dataset)
    train_dataset = shuffle_dataset(train_dataset)

    validation_samples = len(validation_dataset)
    validation_dataset = shuffle_dataset(validation_dataset)

    # Print some useful statistics
    print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(image[1] for image in train_dataset)
    print(train_counters)
    
    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(image[1] for image in validation_dataset)
    print(val_counters)
    print("___________________")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    # Create the data loaders
    validation_labels = np.asarray([row[1] for row in validation_dataset])
    labels = np.asarray([row[1] for row in train_dataset])
    try:
        train_dataset = DeepFakesDataset(np.asarray([row[0] for row in train_dataset], dtype=object), labels, config['model']['image-size'])
        dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                     batch_sampler=None, num_workers=0, collate_fn=None,
                                     pin_memory=False, drop_last=False, timeout=0,
                                     worker_init_fn=None,
                                     persistent_workers=False)
        del train_dataset

        validation_dataset = DeepFakesDataset(np.asarray([row[0] for row in validation_dataset], dtype=object), validation_labels, config['model']['image-size'], mode='validation')
        val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                        batch_sampler=None, num_workers=0, collate_fn=None,
                                        pin_memory=False, drop_last=False, timeout=0,
                                        worker_init_fn=None,
                                        persistent_workers=False)
        del validation_dataset
    except Exception as e:
        print(e)
    

    #model = model.cuda()
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, num_epochs + 1):
        if not_improved_loss == patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*training_params['bs'])+len(val_dl))
        train_correct = 0
        positive = 0
        negative = 0
        try:
            for index, (images, labels) in enumerate(dl):
                #images = torch.unsqueeze(images, dim=0).repeat(training_params['bs'], 1, 1, 1)
                images = np.transpose(images, (0, 3, 1, 2))
                #labels = torch.unsqueeze(labels, dim=0).repeat(training_params['bs'], 1)
                #print(labels)
                labels = labels.unsqueeze(1)
                #images = images.cuda()
                labels = labels.float()
                y_pred = model(images)

                y_pred = y_pred.cpu()

                loss = loss_fn(y_pred, labels)

                corrects, positive_class, negative_class = check_correct(y_pred, labels)
                train_correct += corrects
                positive += positive_class
                negative += negative_class
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
                counter += 1
                total_loss += round(loss.item(), 2)
                for i in range(training_params['bs']):
                    bar.next()


                if index%1200 == 0:
                    print("\nLoss: ", total_loss/counter, "Accuracy: ",train_correct/(counter*training_params['bs']) ,"Train 0s: ", negative, "Train 1s:", positive)


            val_counter = 0
            val_correct = 0
            val_positive = 0
            val_negative = 0

            train_correct /= train_samples
            total_loss /= counter
            for index, (val_images, val_labels) in enumerate(val_dl):

                val_images = np.transpose(val_images, (0, 3, 1, 2))

                #val_images = val_images.cuda()
                val_labels = val_labels.unsqueeze(1)
                val_pred = model(val_images)
                val_pred = val_pred.cpu()
                val_labels = val_labels.float()
                val_loss = loss_fn(val_pred, val_labels)
                total_val_loss += round(val_loss.item(), 2)
                corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
                val_correct += corrects
                val_positive += positive_class
                val_negative += negative_class
                val_counter += 1
                bar.next()

            scheduler.step()
            bar.finish()


            total_val_loss /= val_counter
            val_correct /= validation_samples
            if previous_loss <= total_val_loss:
                print("Validation loss did not improved")
                not_improved_loss += 1
            else:
                not_improved_loss = 0

            previous_loss = total_val_loss
            result_str = "#" + str(t) + "/" + str(num_epochs) + " loss:" + str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(np.count_nonzero(validation_labels == 0)) + " val_1s:" + str(val_positive) + "/" + str(np.count_nonzero(validation_labels == 1))
            print(result_str)
            model_base_name = model_name.split('.')[0]
            global MODELS_PATH
            MODELS_PATH += '/' + model_base_name
            if not os.path.exists(MODELS_PATH):
                os.makedirs(MODELS_PATH)
            torch.save(model.state_dict(), os.path.join(MODELS_PATH, model_name))
            return os.path.join(MODELS_PATH, model_name)
        except Exception as e:
            print(e)

        
