import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
from sklearn import metrics
from scipy.signal import correlate


from cryptopulse.data import data_provider
from cryptopulse.model import CryptoPulse


def calculate_metrics(pred, true):
    mae = metrics.mean_absolute_error(true.flatten(), pred.flatten())
    mse = metrics.mean_squared_error(true.flatten(), pred.flatten())
    corr = correlate(pred.flatten(), true.flatten())
    corr = corr[len(corr) // 2]
    corr = corr / np.sqrt(np.sum(pred**2) * np.sum(true**2))
    return mae, mse, corr


class EarlyStopping(object):
    def __init__(self, patience, tolerance=0):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.best_val_loss = np.inf
        self.tolerance = tolerance

    def __call__(self, cur_val_loss, model, best_model_path):
        if cur_val_loss > self.best_val_loss + self.tolerance:  # no improvement
            self.counter += 1
            print(
                " " * 4,
                "Early stopping counter: {} out of {}".format(
                    self.counter, self.patience
                ),
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # has improvement
            torch.save(model.state_dict(), best_model_path)  # save model
            self.counter = 0  # reset counter
            if cur_val_loss < self.best_val_loss:  # new best loss
                self.best_val_loss = cur_val_loss
                print(
                    " " * 4,
                    (
                        "Validation loss decreased ({:.6f} --> {:.6f}). "
                        "Saving model ..."
                    ).format(self.best_val_loss, cur_val_loss),
                )


def adjust_learning_rate(optimizer, epoch, start_learning_rate, strategy):
    lr_adjust = {epoch: start_learning_rate * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(" " * 4, "Learning rate updated to {}".format(lr))


class Workflow(object):
    def __init__(self, configs):
        super(Workflow, self).__init__()

        self.configs = configs
        # used to save records of this workflow
        self.signature = "{}".format(configs.data)

        self.tr_iter_to_print = 100
        self.te_iter_to_print = 20
        self.tr_print_counter = 0
        # folder path to save results
        self.result_folder = Path("./results") / configs.exp_name / self.signature
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = CryptoPulse(self.configs).float().to(self.device)

    def get_data(self, dev_stage):
        choices = {"train", "dev", "test"}
        if dev_stage not in choices:
            raise ValueError("dev_stage must be one of {}", choices)
        return data_provider(self.configs, dev_stage)

    def setup_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)

    def setup_criterion(self):
        return nn.MSELoss()

    def _tr_print_status(self, step, epoch, loss):
        time_now = time.perf_counter()
        self.tr_print_counter += 1

        speed = (time_now - self.train_start_time) / self.tr_print_counter
        left_time_this_epoch = (
            speed * (self.train_steps_per_epoch - step) / self.tr_iter_to_print
        )
        num_prints = self.train_steps_per_epoch / self.tr_iter_to_print
        left_time_oth_epochs = speed * (
            (self.configs.train_epochs - epoch) * num_prints
        )
        print(
            " " * 4 * 2,
            (
                "Steps: {}, Epoch: {}/{} | Train Loss: {:.4f} "
                "Speed: {:.2f}s/{} steps | Left Time: {}s"
            ).format(
                step,
                epoch,
                self.configs.train_epochs,
                np.average(loss[-self.tr_iter_to_print :]),
                speed,
                self.tr_iter_to_print,
                int(left_time_this_epoch + left_time_oth_epochs),
            ),
        )

    def _load_minibatch_to_device(self, batch):
        return tuple(data_.float().to(self.device) for data_ in batch)

    def _run_model_on_one_batch(self, x_data, y_data):
        preds = self.model(x_data)
        preds = preds[:, -self.configs.pred_len :, -1:]
        trues = y_data[:, -self.configs.pred_len :, -1:]
        return preds, trues

    def train(self, exp_i):
        train_loader = self.get_data("train")
        dev_loader = self.get_data("dev")
        # places to save checkpoints
        checkpoint_path = Path("./checkpoints") / self.signature
        if not checkpoint_path.exists():
            checkpoint_path.mkdir(parents=True)
        best_model_path = checkpoint_path / "checkpoint_exp{}.pth".format(exp_i)
        # training setting
        early_stopping = EarlyStopping(patience=self.configs.patience)
        model_optim = self.setup_optimizer()
        criterion = self.setup_criterion()
        # training
        self.train_steps_per_epoch = len(train_loader)
        self.train_start_time = time.perf_counter()
        for epoch in range(1, self.configs.train_epochs + 1):
            avg_train_loss = []
            self.model.train()
            for step, batch in enumerate(train_loader, start=1):
                model_optim.zero_grad()
                x_data, y_data = self._load_minibatch_to_device(batch)
                preds, trues = self._run_model_on_one_batch(x_data, y_data)
                loss = criterion(preds, trues)
                avg_train_loss.append(loss.item())
                if step % self.tr_iter_to_print == 0:
                    self._tr_print_status(step, epoch, avg_train_loss)
                loss.backward()
                model_optim.step()
            # record changes of dev and test loss per epoch
            avg_train_loss = np.average(avg_train_loss)
            avg_val_loss = self.validation(dev_loader, criterion)
            print(
                " " * 4,
                (
                    "Epoch: {}/{}, Steps: {} | Train Loss: {:.4f} | "
                    "Validation Loss: {:.4f}"
                ).format(
                    epoch,
                    self.configs.train_epochs,
                    step,
                    avg_train_loss,
                    avg_val_loss,
                ),
            )
            # check early stopping per epoch, and save current model
            early_stopping(avg_val_loss, self.model, best_model_path)
            if early_stopping.early_stop:
                print(" " * 4, "Early stopping")
                break
            # adjust learning rate per epoch
            adjust_learning_rate(
                model_optim, epoch, self.configs.learning_rate, self.configs.lr_adj
            )
        # load the best model as the trained model
        self.model.load_state_dict(torch.load(best_model_path, weights_only=False))
        return self.model

    def validation(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(vali_loader, start=1):
                # load minibatch
                x_data, y_data = self._load_minibatch_to_device(batch)
                # run model
                preds, trues = self._run_model_on_one_batch(x_data, y_data)
                # evaluaton
                loss = criterion(preds.cpu(), trues.cpu())
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def _te_evaluation(self, preds, trues, test_folder_path):
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, corr = calculate_metrics(preds, trues)
        print(
            " " * 4,
            ("mae: {:.4f} | mse: {:.4f} | corr: {:.4f}").format(mae, mse, corr),
        )
        with (test_folder_path / "result.txt").open("a") as ofp:
            ofp.write(self.signature + "  \n")
            ofp.write("mae: {:.4f} | mse: {:.4f} corr_x: {}".format(mae, mse, corr))
            ofp.write("\n\n")
        np.save(test_folder_path / "pred.npy", preds)
        res_df = pd.DataFrame(
            [
                [
                    mae,
                    mse,
                    corr,
                    self.signature,
                    int(test_folder_path.stem.split("_")[-1]),
                ]
            ],
            columns=[
                "mae",
                "mse",
                "corr",
                "model",
                "exp_no",
            ],
        )
        return res_df

    def test(self, exp_i):
        self.model.eval()
        test_folder_path = self.result_folder / "experiment_{}".format(exp_i)
        if not test_folder_path.exists():
            test_folder_path.mkdir(parents=True)
        preds, trues = [], []
        with torch.no_grad():
            test_loader = self.get_data("test")
            for step, batch in enumerate(test_loader, start=1):
                x_data, y_data = self._load_minibatch_to_device(batch)
                pred, true = self._run_model_on_one_batch(x_data, y_data)
                pred = pred.cpu().numpy()
                true = true.cpu().numpy()
                preds.append(pred)
                trues.append(true)
        return self._te_evaluation(preds, trues, test_folder_path)
