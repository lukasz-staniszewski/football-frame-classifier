from data_loader.data_loaders import TestDataLoader
from model.model import ResNetClassifier
import torch
import json
from utils import prepare_device
import pandas as pd


def load_model_device(model_path):
    with open("config.json", "r") as f:
        n_gpu = json.load(f)["n_gpu"]
    device, device_ids = prepare_device(n_gpu)
    model = ResNetClassifier()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model, device


def load_dl():
    with open("config.json", "r") as f:
        config = json.load(f)
        test_folder = config["predicting"]["test_folder"]
        config_dl = config["data_loader"]["args"]
    loader = TestDataLoader(
        images_folder=test_folder,
        batch_size=config_dl["batch_size"],
        shuffle=False,
        validation_split=0.0,
        num_workers=config_dl["num_workers"],
    )
    return loader


def perform_predicts():
    with open("config.json", "r") as f:
        config = json.load(f)
        model_path = config["model_to_load_path"]
        results_path = config["predicting"]["results_file"]
    model, device = load_model_device(model_path)
    loader = load_dl()

    preds = torch.Tensor()
    probs = torch.Tensor()
    paths = []

    with torch.no_grad():
        for data in loader:
            image, path = data
            image = image.to(device)
            paths += list(path)
            outputs = model(image).cpu()
            prob, pred = torch.max(outputs, 1)
            preds = torch.cat((preds, pred))
            probs = torch.cat((probs, prob))

    predictions_df = pd.DataFrame(
        {
            "filename": paths,
            "category": map(lambda x: loader.index2class[x.item()], preds.int()),
            "probability": probs,
        }
    )
    predictions_df.to_csv(results_path)
    print("Predictions distribution:")
    print(predictions_df["category"].value_counts())
    print("Predictions performed to file: " + results_path)


def main():
    perform_predicts()


if __name__ == "__main__":
    main()
