from dataload.dataset import CMEMS
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model.SVRNN_module import *


def train(model, args):
    # log
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    summary_path = os.path.join(".", "log",  TIMESTAMP)
    os.makedirs(summary_path, exist_ok=True)
    writer = SummaryWriter(log_dir=summary_path)

    # DataSet
    ds_train = CMEMS(args, 'train')
    ds_val = CMEMS(args, 'val')
    ds_max, ds_min = ds_train.find_max_min()
    mask = ds_train.mask_data()
    mask = torch.tensor(mask.values).to(device)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(ds_val,     batch_size=args.batch_size, shuffle=False,drop_last=True, num_workers=args.num_workers)

    model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    print('\nStart of training')
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        t = tqdm(train_loader, leave=False, ncols=100, total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (inputs, targets) in enumerate(t):
            inputs, targets= inputs.to(device), targets.to(device)
            loss = model_train_loss(model, inputs, targets, mask, ds_max, ds_min)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=loss.item())
            train_loss.append(loss)
        train_loss = torch.mean(torch.stack(train_loss))
        writer.add_scalar('Train/Loss', train_loss, global_step=epoch+1)

        # Vid
        model.eval()
        with torch.no_grad():
            static_loss, masked_loss  = [], []
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                val_m = model_val_loss(model, inputs, targets, mask, ds_max, ds_min)
                masked_loss.append(val_m)
            masked_loss = torch.stack(masked_loss)
            mse_m_loss, mae_m_loss, rmse_m_loss, mape_m_loss, r2_m_loss = masked_loss.mean(dim=0)

        #log
        writer.add_scalar('Val/MSE_m', mse_m_loss, global_step=epoch + 1)
        writer.add_scalar('Val/MAE_m', mae_m_loss, global_step=epoch + 1)
        writer.add_scalar('Val/RMSE_m', rmse_m_loss, global_step=epoch + 1)
        writer.add_scalar('Val/MAPE_m', mape_m_loss, global_step=epoch + 1)
        writer.add_scalar('Val/R2_m', r2_m_loss, global_step=epoch + 1)

        create_directory(args.model_save_path)
        create_directory(args.check_save_path)

        if (epoch + 1) % args.check_seq_len == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'{args.check_save_path}/checkpoint_{epoch + 1}.pth')
            print(f'Saved checkpoint for epoch {epoch + 1}')

        if epoch == 0:
            best_val = float('inf')

        if mse_m_loss < best_val:
            best_val = mse_m_loss
            torch.save(model.state_dict(), f'{args.model_save_path}/best_model-{mse_m_loss:.5f}.pth')


def test(model, args):

    test_file =find_latest_file(args.model_save_path)
    # load data
    ds_test = CMEMS(args, 'test')
    mask = ds_test.mask_data()
    mask = torch.tensor(mask.values).to(device)
    test_max, test_min = ds_test.find_max_min()
    test_loader = DataLoader(ds_test, batch_size=args.test_batch_size, shuffle=False, drop_last=True)

    model.to(device)
    model.load_state_dict(torch.load(f'{args.model_save_path}/{test_file}'))
    model.eval()
    with torch.no_grad():
        static_loss, masked_loss = [], []
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            val_m = model_val_loss(model, inputs, targets, mask, test_max, test_min)
            masked_loss.append(val_m)
        masked_loss = torch.stack(masked_loss)
        Mse_m_loss, Mae_m_loss, Rmse_m_loss, Mape_m_loss, R2_m_loss = masked_loss.mean(dim=0)



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from config import args
    args = args()

    from SVRNN_Github.model.SVRNN import SVRNN_Model
    model = SVRNN_Model(configs=args).to(device)

    train(model, args)
    test(model, args)
