import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
from sklearn import metrics
import pickle

class Experiment():
    def __init__(self, config=None, config_id=None, k=0, model=None, epoch_range=(0,100), last_global_step=0, train_loader=None, 
            val_loader=None, device=None, criterion=torch.nn.BCEWithLogitsLoss(), optimizer=None, stopper=None, scheduler=None, 
            writer=None, model_dir=None, checkpoint_dir=None, start_metric=-1, longitudinal=True) -> None:
        super().__init__()
        self.config = config
        self.config_id  = config_id
        self.kfold = k
        self.model = model
        self.epoch_range = epoch_range
        self.global_step = last_global_step
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.stopper = stopper
        self.scheduler = scheduler
        self.writer = writer
        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = start_metric # large metric is loss, small if metric is acc
        self.longitudinal = longitudinal # True if longitudinal data

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        best_metric_epoch = 0
        start_epoch, epochs = self.epoch_range
        for epoch in range(start_epoch, epochs):
            print(f"Fold {self.kfold}, Epoch {epoch} =============")
            step = 0

            for batch in tqdm(self.train_loader):
                self.model.train()
                imgs, codes, label = (
                    batch["imgs"].to(self.device),
                    batch["ehr"].to(self.device),
                    batch["label"].to(self.device),
                )
                if self.longitudinal:
                    padding = batch['padding'].to(self.device)
                    times = batch["times"].to(self.device)
                    output = self.model(imgs, codes, padding, times)
                else:
                    output = self.model(imgs, codes)
                    # output = output[:,0] # collapse the channel dim
                label = F.one_hot(label, num_classes=2).to(torch.float32)
                loss = self.criterion(output, label)

                # check if loss becomes nan
                if math.isnan(loss):
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"isnan_step{self.global_step}.tar")
                    torch.save({
                        'step': self.global_step,
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_metric': self.best_metric,
                    }, checkpoint_path)
                    print(f'Saved NaN state at step {self.global_step}')
                    return

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                acc = (output.argmax(dim=1) == label.argmax(dim=1)).float().mean()
                # epoch_acc += acc / len(train_loader)
                # epoch_loss += loss / len(train_loader)

                self.writer.add_scalar("Acc/train", acc, self.global_step)
                self.writer.add_scalar("Loss/train", loss, self.global_step)

                # Validation
                if self.global_step % self.config["val_interval"] == 0:
                    val_loss, val_acc = self.validate()
                    
                    # update stopping criteria every val period
                    self.stopper.step(val_loss.cpu().numpy())
                    
                    if val_loss < self.best_metric:
                        self.best_metric = val_loss
                        best_metric_epoch = epoch
                        model_path = os.path.join(self.model_dir, f"best_model.pth")
                        torch.save(self.model.state_dict(), model_path)
                        print("Saved new best model")
                    print(f"{self.config_id}:", f"\n Current: epoch {epoch}, mean acc {val_acc.item():.4f}", 
                        f"\n Best: epoch {best_metric_epoch}, mean loss {self.best_metric.item():.4f}")
                
                # Early stopping
                if self.stopper.loss_check_stop():
                    return
                # Stop if val loss is close to zero
                if val_loss < 1e-3:
                    return
                
                step += 1
                self.global_step += 1
                
            # Save checkpoints
            if epoch % self.config["checkpoint_interval"] == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch{epoch}.tar")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_metric': self.best_metric,
                }, checkpoint_path)
    
    def validate(self):
    # def validate(model, val_loader, device, criterion, writer, global_step):
        self.model.eval()
        val_acc = 0
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                imgs, codes, label = (
                    batch["imgs"].to(self.device),
                    batch["ehr"].to(self.device),
                    batch["label"].to(self.device),
                )
                if self.longitudinal:
                    padding = batch['padding'].to(self.device)
                    times = batch["times"].to(self.device)
                    output = self.model(imgs, codes, padding, times)
                else:
                    output = self.model(imgs, codes)
                label = F.one_hot(label, num_classes=2).to(torch.float32)
                loss = self.criterion(output, label)

                acc = (output.argmax(dim=1) == label.argmax(dim=1)).float().mean()
                val_acc += acc
                val_loss += loss
                
        val_acc /= len(self.val_loader)
        val_loss /= len(self.val_loader)
        self.writer.add_scalar('Acc/val', val_acc, self.global_step)
        self.writer.add_scalar('Loss/val', val_loss, self.global_step)
        return val_loss, val_acc

    def test(self):
        self.model.eval()
        metrics_dict = {}
        outputs = np.zeros(len(self.val_loader))
        labels = np.zeros(len(self.val_loader), dtype=np.uint8)

        df_rows = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.val_loader)):
                fnames, imgs, codes, label = (
                    batch["fnames"],
                    batch["imgs"].to(self.device),
                    batch["ehr"].to(self.device),
                    batch["label"],
                )
                if self.longitudinal:
                    padding = batch['padding'].to(self.device)
                    times = batch["times"].to(self.device)
                    output = self.model(imgs, codes, padding, times)
                else:
                    output = self.model(imgs, codes)

                output = torch.sigmoid(output)[0,1] # score for class 1
                outputs[i] = output.cpu().numpy()
                labels[i] = label.numpy()

                pid = fnames[0][0].split('time')[0]
                df_rows.append({'pid': pid, 'lung_cancer': label.numpy()[0], 'pred': output.cpu().numpy()})

        df = pd.DataFrame(df_rows)
        df.to_csv(os.path.join(self.model_dir, "pred.csv"))

        fpr, tpr, _ = metrics.roc_curve(labels, outputs)
        roc_auc = metrics.auc(fpr, tpr)
        metrics_dict["roc_auc"], metrics_dict["fpr"], metrics_dict["tpr"] = roc_auc, fpr, tpr
        print(f"AUC: {roc_auc}")

        metrics_path = os.path.join(self.model_dir, f"metrics.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics_dict, f)

    def pretrain_train(self):
        best_metric_epoch = 0
        start_epoch, epochs = self.epoch_range
        for epoch in range(start_epoch, epochs):
            step = 0

            for batch in tqdm(self.train_loader):
                self.model.train()
                imgs, codes = (
                    batch["imgs"].to(self.device),
                    batch["ehr"].to(self.device),
                )
                if self.longitudinal:
                    times = batch["times"].to(self.device)
                    loss = self.model(imgs, codes, times)
                else:
                    loss = self.model(imgs, codes)

                # check if loss becomes nan
                if math.isnan(loss):
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"isnan_step{self.global_step}.tar")
                    torch.save({
                        'step': self.global_step,
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_metric': self.best_metric,
                    }, checkpoint_path)
                    print(f'Saved NaN state at step {self.global_step}')
                    return

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self.writer.add_scalar("Loss/train", loss, self.global_step)

                # Validation
                if self.global_step % self.config["val_interval"] == 0:
                    val_loss = self.pretrain_validate()
                    
                    # update stopping criteria every val period
                    self.stopper.step(val_loss.cpu().numpy())
                    
                    if val_loss < self.best_metric:
                        self.best_metric = val_loss
                        best_metric_epoch = epoch
                        mae_path = os.path.join(self.model_dir, "best_mae.pth")
                        encoder_path = os.path.join(self.model_dir, "best_encoder.pth")
                        torch.save(self.model.state_dict(), mae_path)
                        torch.save(self.model.encoder.state_dict(), encoder_path)
                        print("Saved new best model")
                    print(f"{self.config_id}:"
                        f"\n Current: epoch {epoch}, mean loss {val_loss.item():.4f}"
                    f"\n Best: epoch {best_metric_epoch}, mean loss {self.best_metric.item():.4f}")
                
                # Early stopping
                if self.stopper.loss_check_stop():
                    return
                
                step += 1
                self.global_step += 1
                
            # Save checkpoints
            if epoch % self.config["checkpoint_interval"] == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch{epoch}.tar")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_metric': self.best_metric,
                }, checkpoint_path)

    def pretrain_validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
            # batch = next(val_iterator)
                imgs, codes = (
                    batch["imgs"].to(self.device),
                    batch["ehr"].to(self.device),
                )
                if self.longitudinal:
                    times = batch["times"].to(self.device)
                    loss = self.model(imgs, codes, times)
                else:
                    loss = self.model(imgs, codes)

                val_loss += loss
                
        val_loss /= len(self.val_loader)
        self.writer.add_scalar('Loss/val', val_loss, self.global_step)
        return val_loss