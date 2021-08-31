import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model import metric
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))  # 로그스텝?
        #self.log_step = 100
        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        label_name = self.config['arch']['args']['label_name']
        for batch_idx, (data, target, gender, age, mask) in enumerate(self.data_loader):

            data = data.to(self.device)
            if label_name == 'gender':
                gender = gender.to(self.device)
            elif label_name == 'age':
                age = age.to(self.device)
            elif label_name == 'mask':
                mask = mask.to(self.device)
            elif label_name == 'total':
                target = target.to(self.device)

# age filtering 57
# 0.47 vs 0.33 vs 0.2

            self.optimizer.zero_grad()
            output = self.model(data)
            if label_name == 'gender':
                self.criterion = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor([1.5, 1.0]).to(self.device))
                loss = self.criterion(output, gender)
            elif label_name == 'age':
                self.criterion = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor([2., 3.3, 4.7]).to(self.device))
                loss = self.criterion(output, age)
            elif label_name == 'mask':
                self.criterion = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor([1., 2., 2.]).to(self.device))
                loss = self.criterion(output, mask)
            elif label_name == 'total':
                loss = self.criterion(output, mask)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                if label_name == 'gender':
                    self.train_metrics.update(
                        met.__name__, met(output, gender))
                elif label_name == 'age':
                    self.train_metrics.update(met.__name__, met(output, age))
                elif label_name == 'mask':
                    self.train_metrics.update(met.__name__, met(output, mask))
                elif label_name == 'total':
                    self.train_metrics.update(
                        met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(
                    data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        # full_output = torch.tensor([]).cuda()
        # full_target = torch.tensor([]).cuda()
        label_name = self.config['arch']['args']['label_name']
        with torch.no_grad():
            for batch_idx, (data, target, gender, age, mask) in enumerate(self.valid_data_loader):
                data = data.to(self.device)
                if label_name == 'gender':
                    gender = gender.to(self.device)
                    # full_target = torch.cat([full_target, gender])
                elif label_name == 'age':
                    age = age.to(self.device)
                    # full_target = torch.cat([full_target, age])
                elif label_name == 'mask':
                    mask = mask.to(self.device)
                    # full_target = torch.cat([full_target, mask])
                elif label_name == 'total':
                    target = target.to(self.device)
                    # full_target = torch.cat([full_target, target])

                output = self.model(data)
                # full_output = torch.cat([full_output, output])
                if label_name == 'gender':
                    loss = self.criterion(output, gender)
                elif label_name == 'age':
                    loss = self.criterion(output, age)
                elif label_name == 'mask':
                    loss = self.criterion(output, mask)
                elif label_name == 'total':
                    loss = self.criterion(output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    # if met.__name__=='f1':
                    #     continue
                    if label_name == 'gender':
                        self.valid_metrics.update(
                            met.__name__, met(output, gender))
                    elif label_name == 'age':
                        self.valid_metrics.update(
                            met.__name__, met(output, age))
                    elif label_name == 'mask':
                        self.valid_metrics.update(
                            met.__name__, met(output, mask))
                    elif label_name == 'total':
                        self.valid_metrics.update(
                            met.__name__, met(output, target))

                self.writer.add_image('input', make_grid(
                    data.cpu(), nrow=8, normalize=True))

                if label_name == 'gender':
                    target = gender
                elif label_name == 'age':
                    target = age
                elif label_name == 'mask':
                    target = mask
                elif label_name == 'total':
                    target = target

                conf_mat = confusion_matrix(
                    target.cpu(), torch.argmax(output, dim=1).cpu())
                fig = plt.figure()
                sns.heatmap(conf_mat, cmap=plt.cm.Blues, annot=True, fmt='g')
                plt.show()
                plt.xlabel('predicted label')
                plt.ylabel('true label')
                self.writer.add_figure('fig', fig)
        # self.valid_metrics.update('f1', metric.f1(full_output, full_target))
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
