import os
import torch


def create_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        print(f"创建目录失败: {e}")


def find_latest_file(directory):
    latest_file = None
    latest_mtime = None
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            mtime = os.path.getmtime(filepath)
            if latest_mtime is None or mtime > latest_mtime:
                latest_file = filename
                latest_mtime = mtime
    if not latest_file:
        return None
    return latest_file


def model_train_loss(model, inputs, targets, mask, max_train, min_train):
    device = inputs.device
    mask = mask.to(device).unsqueeze(-1)
    total = torch.cat((inputs, targets), dim=1).to(device)
    outputs = model(total)

    scale = max_train - min_train
    total = total * scale + min_train
    outputs = outputs * scale + min_train

    total = total[:, 1:]
    B, T, _, _, _ = outputs.shape
    valid_elements = mask.sum() * B * T
    pre_mask = outputs * mask
    tar_mask = total * mask

    train_loss = torch.sum((pre_mask - tar_mask) ** 2) / valid_elements

    return train_loss


def model_val_loss(model, inputs, targets, mask, max_val, min_val):
    device = inputs.device
    mask = mask.to(device).unsqueeze(-1)
    total = torch.cat((inputs, targets), dim=1).to(device)
    outputs = model(total)[:, 11:]

    scale = max_val - min_val
    targets = targets * scale + min_val
    outputs = outputs * scale + min_val

    pre_valid = outputs * mask
    tar_valid = targets * mask
    diff = pre_valid - tar_valid

    abs_diff = torch.abs(diff)
    squared_diff = diff ** 2
    B, T, _, _, _ = outputs.shape
    valid_elements = mask.sum() * B * T

    MSE_m = torch.sum(squared_diff) / valid_elements
    MAE_m = torch.sum(abs_diff) / valid_elements
    RMSE_m = torch.sqrt(MSE_m)
    MAPE_m = torch.sum(abs_diff / (tar_valid + 1e-8)) / valid_elements
    tar_mean_m = torch.sum(tar_valid) / valid_elements
    R2_m = 1 - torch.sum(squared_diff) / torch.sum((tar_valid - tar_mean_m) ** 2)

    val_m = torch.tensor([MSE_m, MAE_m, RMSE_m, MAPE_m, R2_m])
    return val_m