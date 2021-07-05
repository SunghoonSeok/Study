"""Trainer 클래스 정의

TODO:

NOTES:

REFERENCE:

UPDATED:
"""

import os
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error

import torch


class CustomTrainer():

    """ CustomTrainer
        epoch에 대한 학습 및 검증 절차 정의
    
    Attributes:
        model (`model`)
        device (str)
        loss_fn (Callable)
        metric_fn (Callable)
        optimizer (`optimizer`)
        scheduler (`scheduler`)
        logger (`logger`)
    """

    def __init__(self, model, device, loss_fn, metric_fn, optimizer=None, scheduler=None, logger=None):
        """ 초기화
        """
        self.model = model
        self.device = device
        self.logger = logger
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # History - loss
        self.train_batch_loss_mean_list = list()
        self.train_batch_score_list = list()

        self.validation_batch_loss_mean_list = list()
        self.validation_batch_score_list = list()

        # History - predict
        self.train_target_pred_list = list()
        self.train_target_list = list()

        self.validation_target_pred_list = list()
        self.validation_target_list = list()

        # History - answer
        self.train_answer_list = list()
        self.validation_answer_list = list()

        # Output
        self.train_loss_mean = 0
        self.train_loss_sum = 0
        self.train_score = 0

        self.validation_loss_mean = 0
        self.validation_loss_sum = 0
        self.validation_score = 0

        # Prediction
        # self.prediction_target_list = list()
        # self.prediction_target_pred_list = list()
        self.prediction_score_list = list()
        self.answer_list = list()


    def train_epoch(self, dataloader, epoch_index=0, verbose=False, logging_interval=1):
        """ 한 epoch에서 수행되는 학습 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
            verbose (boolean)
            logging_interval (int)
        """
        self.model.train()

        for batch_index, data in enumerate(dataloader):

            feature = data['feature'].to(self.device)
            answer = data['target']

            self.optimizer.zero_grad()

            # Loss
            pred_feature = self.model(feature)
            batch_loss_mean = self.loss_fn(pred_feature, feature)
            batch_loss_sum = batch_loss_mean.item() * dataloader.batch_size
            self.train_batch_loss_mean_list.append(batch_loss_mean.item())
            self.train_loss_sum += batch_loss_sum

            # Metric
            batch_score_list = torch.pow(pred_feature - feature, 2).mean(axis=1)
            batch_score_mean = batch_score_list.mean()
            self.train_batch_score_list.append(batch_score_list.cpu().tolist())

            # History - predict
            self.train_target_list.extend(feature.cpu().tolist())
            self.train_target_pred_list.extend(pred_feature.cpu().tolist())
            self.train_answer_list.extend(answer.tolist())

            # Update
            batch_loss_mean.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Log verbose
            if verbose & (batch_index % logging_interval == 0):
                msg = f"Epoch {epoch_index} train batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score_mean}"
                self.logger.info(msg) if self.logger else print(msg)

        self.train_loss_mean = self.train_loss_sum / len(dataloader)
        self.train_score_list = np.mean(np.square([np.array(self.train_target_pred_list) - np.array(self.train_target_list)]), axis=2).squeeze(0)
        # print(len(self.train_score_list))
        # print(len(self.train_answer_list))
        # print(self.train_score_list)
        # print(self.train_answer_list)

        # # auroc_score
        # auroc_score = roc_auc_score(self.train_answer_list, self.train_score_list) # /self.train_score_list.max())
        # print(auroc_score)
        # self.train_score = auroc_score

        mse_score = np.mean(np.array(self.train_batch_score_list))
        self.train_score = 0.5

        msg = f'Epoch {epoch_index}, Train, Mean loss: {self.train_loss_mean}, MSE Score: {mse_score}'
        self.logger.info(msg) if self.logger else print(msg)


    def validate_epoch(self, dataloader, epoch_index=0, verbose=False, logging_interval=1):
        """ 한 epoch에서 수행되는 검증 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.eval()

        with torch.no_grad():
            for batch_index, data in enumerate(dataloader):

                feature = data['feature'].to(self.device)
                answer = data['target']

                self.optimizer.zero_grad()

                # Loss
                pred_feature = self.model(feature)
                batch_loss_mean = self.loss_fn(pred_feature, feature)
                batch_loss_sum = batch_loss_mean.item() * dataloader.batch_size
                self.validation_batch_loss_mean_list.append(batch_loss_mean.item())
                self.validation_loss_sum += batch_loss_sum

                # Metric
                batch_score_list = torch.pow(pred_feature - feature, 2).mean(axis=1)
                batch_score_mean = batch_score_list.mean()
                self.validation_batch_score_list.append(batch_score_list.cpu().tolist())

                # History - predict
                self.validation_target_list.extend(feature.cpu().tolist())
                self.validation_target_pred_list.extend(pred_feature.cpu().tolist())
                self.validation_answer_list.extend(answer.tolist())

                # Log verbose
                if verbose & (batch_index % logging_interval == 0):
                    msg = f"Epoch {epoch_index} validation batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score_mean}"
                    self.logger.info(msg) if self.logger else print(msg)

            self.validation_loss_mean = self.validation_loss_sum / len(dataloader)
            self.validation_score_list = np.mean(np.square([np.array(self.validation_target_pred_list) - np.array(self.validation_target_list)]), axis=2).squeeze(0)
            # print(len(validation_score_list))
            # print(len(validation_answer_list))

            auroc_score = roc_auc_score(self.validation_answer_list, self.validation_score_list/self.validation_score_list.max())
            print(auroc_score)
            self.validation_score = auroc_score
            msg = f'Epoch {epoch_index}, Validation, Mean loss: {self.validation_loss_mean}, AUROC Score: {auroc_score}'
            self.logger.info(msg) if self.logger else print(msg)


    def predict_epoch(self, dataloader, epoch_index=0, verbose=False, logging_interval=1):
        """ 추론 함수

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.eval()

        with torch.no_grad():
            for batch_index, data in enumerate(dataloader):

                feature = data['feature'].to(self.device)
                answer = data['target']
                pred_feature = self.model(feature)

                target_list = feature.cpu().tolist()
                target_pred_list = pred_feature.cpu().tolist()

                # Get Score
                element_score_list = [self.metric_fn(x[0], x[1]) for x in zip(target_list, target_pred_list)]
                self.prediction_score_list.extend(element_score_list)
                self.answer_list.extend(answer)

    @staticmethod
    def anomaly_score(metric_score):
        return np.exp(metric_score)


    def clear_history(self):
        """ 한 epoch 종료 후 history 초기화
            Examples:
                >>for epoch_index in tqdm(range(EPOCH)):
                >>    trainer.train_epoch(dataloader=train_dataloader, epoch_index=epoch_index, verbose=False)
                >>    trainer.validate_epoch(dataloader=validation_dataloader, epoch_index=epoch_index, verbose=False)
                >>    trainer.clear_history()
        """

        # History - loss
        self.train_batch_loss_mean_list = list()
        self.train_batch_score_list = list()

        self.validation_batch_loss_mean_list = list()
        self.validation_batch_score_list = list()

        # History - predict
        self.train_target_pred_list = list()
        self.train_target_list = list()

        self.validation_target_pred_list = list()
        self.validation_target_list = list()

        # History - answer
        self.train_answer_list = list()
        self.validation_answer_list = list()

        # Output
        self.train_loss_mean = 0
        self.train_loss_sum = 0
        self.train_score = 0

        self.validation_loss_mean = 0
        self.validation_loss_sum = 0
        self.validation_score = 0

        # Prediction
        # self.prediction_target_list = list()
        # self.prediction_target_pred_list = list()
        self.prediction_score_list = list()
        self.answer_list = list()

