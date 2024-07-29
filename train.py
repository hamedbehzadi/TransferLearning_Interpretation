from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import device, flatten, nn, optim
from torch.nn.modules.module import Module

from TransferLearningInterpretation.config import wandb
from TransferLearningInterpretation.dataset import Data
from TransferLearningInterpretation.utils import create_if_not_exist

import pickle
from matplotlib import pyplot as plt

print('trin.py imported')
def save(model, file_path):
    torch.save(model.state_dict(), file_path)


def load(model, file_path, cfg) -> None:
    model.load_state_dict(torch.load(file_path, map_location=cfg.DEVICE_STR))
    model.eval()


class Train(object):
    def __init__(self, model, data, cfg):
        super().__init__()
        self.device = cfg.DEVICE
        self.model = model.to(self.device)
        self.data = data
        self.loss = F.cross_entropy
        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.LEARNING_RATE)
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.train_acc_history = []

    def fit(self, epochs, checkpoints, output_path):

        epoch = 0
        early_stop = False

        wandb.watch(self.model)

        while (epoch != epochs and not early_stop):
            epoch += 1

            # Log start of epoch
            start = datetime.now()
            print(f"Epoch {epoch} started at {start}")

            running_loss = 0.
            train_correct = 0
            train_size = 0
            # Set model to training mode
            self.model.train(True)
            for (inputs, labels) in self.data.train_loader:
                # Move data to GPU
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                loss, t_pred= self.loss_batch(inputs, labels)

                running_loss += loss
                train_correct += (t_pred == labels).float().sum().item()
                train_size += inputs.size(0)

            self.train_loss_history.append(
                running_loss.item()/len(self.data.train_loader))

            self.train_acc_history.append(100 * train_correct / train_size)

            self.model.train(False)
            with torch.no_grad():
                running_vloss = 0.
                correct = 0
                te_size = 0
                for (v_inputs, v_labels) in self.data.test_loader:
                    # Move data to GPU
                    v_inputs = v_inputs.to(self.device)
                    v_labels = v_labels.to(self.device)

                    v_outputs = self.model(v_inputs)
                    _, v_pred = v_outputs.max(dim=1)

                    correct += (v_pred == v_labels).float().sum().item()
                    te_size += v_inputs.size(0)
                    running_vloss += self.loss(v_outputs, v_labels)

                self.val_loss_history.append(
                    running_vloss.item() / len(self.data.test_loader))
                self.val_acc_history.append(100 * correct / te_size)

            wandb.log(
                {"training_loss": self.train_loss_history[-1], "validation_loss": self.val_loss_history[-1],
                 "accuracy": self.val_acc_history[-1]})

            stop = datetime.now()
            print(
                f"Epoch {epoch} ended at {stop}, epoch lasted for {stop-start}")

            if epoch in checkpoints:
                print(f"Saved epoch {epoch} to file.")
                save(self.model,
                     f"{output_path}epoch{(epoch):02}.pth")

            # min_loss = min(self.val_loss_history)
            # if self.val_loss_history[-1] == min_loss:
            #     save(self.model, f"{output_path}best_loss.pth")
            #     continue

            # early_stop = True
            # for i in range(2, 9):
            #     if i > len(self.val_loss_history):
            #         break
            #     if self.val_loss_history[-i] == min_loss:
            #         early_stop = False
            #         break
        with open(output_path + "performance_results.npy", 'wb') as file_add:
            pickle.dump((self.train_loss_history,self.train_acc_history,self.val_loss_history,self.val_acc_history),
                        file_add)
        with open(output_path + "backup_history.txt", "w") as backup_file:
            backup_file.write(f"Training loss: {', '.join([f'{i:0.4f}' for i in self.train_loss_history])}")
            backup_file.write(f"Validation loss: {', '.join([f'{i:0.4f}' for i in self.val_loss_history])}")
            backup_file.write(f"Validation accuracy: {', '.join([f'{i:0.4f}' for i in self.val_acc_history])}")
            backup_file.write(f"Train accuracy: {', '.join([f'{i:0.4f}' for i in self.train_acc_history])}")

        x = np.arange(len(self.train_loss_history))
        plt.plot(x, self.train_loss_history,'r.-',label='Train Loss')
        plt.plot(x, self.val_loss_history,'b.-',label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(output_path+'loss.png')

        plt.cla()

        plt.plot(x, self.train_acc_history, 'r.-', label='Train Acc')
        plt.plot(x, self.val_acc_history, 'b.-', label='Val ACC')
        plt.xlabel('Epoch')
        plt.ylabel('ACC')
        plt.legend()
        plt.savefig(output_path + 'ACC.png')

    def loss_batch(self, inputs, labels):
        # Zero gradients for batch
        self.optim.zero_grad()

        outputs = self.model(inputs)
        _, t_pred = outputs.max(dim=1)
        loss = self.loss(outputs, labels)
        loss.backward()
        # # NaN problem fix: https://discuss.pytorch.org/t/crossentropyloss-loss-becomes-nan-after-several-iteration/73530/2
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optim.step()
        return loss, t_pred


class Validation(object):
    def __init__(self, data: Data, device: device = device("cpu")) -> None:
        super().__init__()
        self.device = device
        self.data = data
        self.model = cfg.BASE_MODEL

    def fit(self):
        #for epoch in ['01','03','05','07','10', '15']:
        for epoch in ['15']:
            model_address = cfg.OUTPUT_PATH + '/' + cfg.MODEL_NAME + '/epoch' + epoch + '.pth'
            self.model.load_state_dict(torch.load(model_address))
            self.model = self.model.to(self.device)
            correct = 0
            te_size = 0
            for (inputs, labels) in self.data.train_loader:
                # Move data to GPU
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, pred = outputs.max(dim=1)

                correct += (pred == labels).float().sum().item()
                te_size += inputs.size(0)

            print(epoch, 100 * (correct / te_size))



