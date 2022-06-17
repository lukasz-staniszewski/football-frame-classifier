
import torch
import json
import pandas as pd
import os
import pathlib
import sys

sys.path.append("..")

from data_loader.data_loaders import OneImageDataLoader
from model.model import ResNetClassifier
from utils import prepare_device



def load_model_device():
    with open("./../config.json", "r") as f:
        json_f = json.load(f)
        n_gpu = json_f["n_gpu"]
        model_to_load_path = os.path.join("./../",json_f["model_to_load_path"])
    
    device, device_ids = prepare_device(n_gpu)
    model = ResNetClassifier()
    checkpoint = torch.load(model_to_load_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model, device


def load_dl(image_folder, image_name):
    
    loader = OneImageDataLoader(
        image_folder=image_folder,
        image_name=image_name,
        batch_size=1,
        shuffle=False,
        validation_split=0.0)
    
    return loader


def perform_predicts(model, device, dl):
    with torch.no_grad():
        image = next(iter(dl))
        image = image.to(device)
        outputs = model(image).cpu()
        prob, pred = torch.max(outputs, 1)
    
    return prob, pred, outputs