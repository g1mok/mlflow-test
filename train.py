import os
import boto3
import datetime
from tqdm import tqdm
from dataset import Customdataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from efficientnet_pytorch import EfficientNet
from modules.pytorchtools import EarlyStopping
from modules.utils import *
from test import *

import argparse

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# import logging
# logging.getLogger("mlflow").setLevel(logging.DEBUG)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_split(data_path):
    files_list = os.listdir(data_path)

    cls_form = [file.split('_')[0] for file in files_list]
    x_train, x_re, y_train, y_re = train_test_split(files_list, cls_form, test_size=0.25, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_re, y_re, test_size=0.2, random_state=42)
    return x_train, y_train, x_val, x_test, y_val, y_test

def saveModel(Config, model_name, model, mode):
    filename = Config['Save_path'] + f'{model_name}_best_{mode}.pth'
    torch.save(model.state_dict(), filename)

def main(run):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='train_config.yaml', help='path to base config yaml file.')
    parser.add_argument("--tag", type=str, default='vx.x', help='parameter to tag version')
    opt = parser.parse_args()

    Config, mlflow_tag = set_train_config(load_yaml(opt.config))
    make_directory(Config['Save_path'])

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), Config['Data_path'])

    # DATA LOAD
    x_train, y_train, x_val, x_test, y_val, y_test = data_split(data_path)
    train_set = Customdataset(data_path, x_train, y_train, Config['Img_mode'], Config['Stroke_mode'])
    train_dataloader = DataLoader(train_set, batch_size=Config['Batch_size'])

    val_set = Customdataset(data_path, x_val, y_val, Config['Img_mode'], Config['Stroke_mode'])
    val_dataloader = DataLoader(val_set, batch_size=Config['Batch_size'])

    test_set = Customdataset(data_path, x_test, y_test, Config['Img_mode'], Config['Stroke_mode'])
    test_dataloader = DataLoader(test_set, batch_size=Config['Batch_size'])
    print(f"train: {len(train_dataloader)}, val: {len(val_dataloader)}, test: {len(test_dataloader)}")
    print(f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")

    # MODEL LOAD
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=n_class).to(device)
    model_name = f"{Config['Model']}_{Config['Img_mode']}_{Config['Model_mode']}"
    
    # MODEL COMPILE
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config['Learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    early_stopping = EarlyStopping(patience=Config['Patience'], verbose=True, path=f'{Config["Save_path"]}/es_{model_name}.pth')
    
    best_accuracy = 0
    lowest_loss = 1

    # TRAIN
    tags = {
        'mlflow.note.content': mlflow_tag['description'],
        'mlflow.user': mlflow_tag['user_name'],
        'mlflow.runName': mlflow_tag['run_name']
    }
    with mlflow.start_run(run_id = run.info.run_id, tags=tags):
        mlflow.log_params(Config)
        mlflow.log_param("DVC_tag", opt.tag)

        for epoch in range(Config['Total_epoch']):
            print(f'{epoch} epoch start! : {datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")}')

            train_loss, train_acc, losses = train_one_epoch(Config, train_dataloader, model, criterion, optimizer)
            valid_loss, valid_acc, epoch_loss = validate(Config, val_dataloader, model, criterion, losses)
            mlflow.log_metrics(
                {"Train_loss":train_loss, 
                "Train_acc": train_acc,
                "Val_loss":valid_loss, 
                "Val_acc": valid_acc,
                "Epoch": epoch}, step=epoch)

            print(f'Epoch {epoch} / {Config["Total_epoch"]}')
            print(f"Train Loss : {train_loss:.4f} // Train Accuracy : {train_acc:.4f}")
            print(f'Val Loss : {valid_loss:.4f} // Val Accuracy : {valid_acc:.4f}')
            print(f'Total_loss : {epoch_loss}')
            scheduler.step(valid_loss)

            if epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                saveModel(Config, model_name, model, 'loss')
                print(f'Saving Lowest loss model_[ epoch : {epoch}, loss : {epoch_loss:.4f}]')

            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                break

            if lowest_loss < 0.00001:
                break

        # log model
        mlflow.pytorch.log_model(
            model, 
            artifact_path='pytorch_model',
            registered_model_name=mlflow_tag['register_model'])
        
        # log state_dict
        state_dict = model.state_dict()
        mlflow.pytorch.log_state_dict(state_dict, artifact_path="checkpoint")

        print(
        "\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), 'pytorch_model')
        )
        print(
        "\nThe state_dict is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), 'checkpoint')
        )

def train_one_epoch(Config, train_loader, model, criterion, optimizer):
    model.train()

    train_losses = 0.0
    train_accuracy = 0
    num_cnt = 0

    losses = AverageMeter()

    for i, (images, strokes, labels) in enumerate(tqdm(train_loader)):

        images = images.to(device)
        labels = labels.to(device)

        if Config['Model_mode'] == 'cnn':
            output = model(images)
        else:
            strokes = strokes.to(device)
            output = model(images, strokes)

        t_loss = criterion(output, labels)
        
        _, preds = torch.max(output, 1)
        labels = torch.argmax(labels, dim=1)

        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()

        train_losses += t_loss.item()
        train_accuracy += (labels == preds).sum().item()
        num_cnt += len(images)
        losses.update(t_loss.item(), images.size(0))
    
    del images, labels, strokes

    train_loss = train_losses / len(train_loader)
    train_acc = train_accuracy / num_cnt
    return train_loss, train_acc, losses

def validate(Config, val_loader, model, criterion, losses):
    valid_losses = 0.0
    valid_accuracy = 0
    num_cnt = 0

    model.eval()
    with torch.no_grad():
        for i, (images, strokes, labels) in enumerate(val_loader):

            images = images.to(device)
            labels = labels.to(device)

            if Config['Model_mode'] == 'cnn':
                output = model(images)
            else:
                strokes = strokes.to(device)
                output = model(images, strokes)

            v_loss = criterion(output, labels)

            _, preds = torch.max(output, 1)
            labels = torch.argmax(labels, dim=1)

            valid_losses += v_loss.item()
            valid_accuracy += (labels == preds).sum().item()
            num_cnt += len(images)
            losses.update(v_loss.item(), images.size(0))

    del images, labels, strokes

    valid_loss = valid_losses / len(val_loader)
    valid_acc = valid_accuracy / num_cnt
    return valid_loss, valid_acc, losses.avg

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://13.125.224.170:8503")
    
    experiment = mlflow.set_experiment('dvc-test')  # set the experiment
    client = MlflowClient()
    run = client.create_run(experiment_id=experiment.experiment_id)
    main(run)