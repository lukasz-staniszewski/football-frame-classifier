from model.model import ResNetClassifier
import torch
import torch
import json
from data_loader.data_loaders import FramesDataLoader
from model import metric
from utils import prepare_device
import pandas as pd
from sklearn.metrics import classification_report


def load_model_device(model_path):
    with open("config.json", "r") as f:
        n_gpu = json.load(f)['n_gpu']
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
        config = json.load(f)['data_loader']['args']
    loader = FramesDataLoader(images_folder=config["images_folder"], batch_size=config["batch_size"], csv_path=config["csv_path"], csv_path_tf=config["csv_path_tf"], shuffle=config["shuffle"], validation_split=config["validation_split"],num_workers=config["num_workers"], is_with_aug=config["is_with_aug"])
    val_loader = loader.split_validation()
    return loader,val_loader


def perform_metrics():
    with open("config.json", "r") as f:
        model_path = json.load(f)['model_to_load_path']
    model,device = load_model_device(model_path)
    loader,val_loader = load_dl()
    
    all_class_names = list(loader.class2index.keys())
    correct_predicts_by_class = {name_class: 0 for name_class in all_class_names}
    predicts_by_class = {name_class: 0 for name_class in all_class_names}
    
    preds = torch.Tensor()
    trgts = torch.Tensor()
    predicted_classes = torch.Tensor()

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            outputs = model(images).cpu()
            _, predicted = torch.max(outputs, 1)
            trgts = torch.cat((trgts, labels))
            preds = torch.cat((preds, outputs))
            predicted_classes = torch.cat((predicted_classes, predicted))
            for label, model_prediction in zip(labels, predicted):
                if label == model_prediction:
                    correct_predicts_by_class[all_class_names[label]] += 1
                predicts_by_class[all_class_names[label]] += 1
    
    print("~~PER CLASS ACCURACIES~~")
    for name_class, count_ok in correct_predicts_by_class.items():
        acc = 100 * count_ok / predicts_by_class[name_class]
        print("Accuracy for {} = {:.2f}%".format(name_class, acc))
    print("~~~~~~~~~~~~~~~~~~~~~~~~")

    print("~~~~~~~~~METRICS~~~~~~~~")
    print("MICRO accuracy:", metric.micro_accuracy(preds, trgts))
    print("MICRO precision:", metric.micro_precision(preds, trgts))
    print("MICRO recall:", metric.micro_recall(preds, trgts))
    print("MICRO f1:", metric.micro_f1(preds, trgts))
    print("MACRO accuracy:", metric.macro_accuracy(preds, trgts))
    print("MACRO precision:", metric.macro_precision(preds, trgts))
    print("MACRO recall:", metric.macro_recall(preds, trgts))
    print("MACRO f1:", metric.macro_f1(preds, trgts))
    print("~~~~~~~~~~~~~~~~~~~~~~~~")

    print("~~~~PREDICTIONS MADE:~~~")
    df = pd.DataFrame(preds.argmax(1)).value_counts()
    index2class = dict((v, k) for k, v in loader.class2index.items())
    df.rename(index=index2class, inplace=True)
    print(df)
    print("~~~~~~~~~~~~~~~~~~~~~~~~")

    print("~CLASSIFICATION REPORT:~")
    print(classification_report(trgts, predicted_classes, target_names=["side_view","closeup","non_match","front_view","side_gate_view","aerial_view","wide_view"], zero_division=0))
    print("~~~~~~~~~~~~~~~~~~~~~~~~")


def main():
    perform_metrics()

if __name__ == "__main__":
    main()