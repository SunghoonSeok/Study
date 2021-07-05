import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
import argparse
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from modules.dataset import Dataset_tu
import albumentations as albu

from modules.utils import load_yaml, save_yaml, get_logger, make_directory

from datetime import datetime, timezone, timedelta
import random

# from modules.dataset import CustomDataset
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder
from modules.trainer import CustomTrainer
from modules.utils import load_yaml, save_yaml, get_logger, make_directory

iou_thres = 0.75

# DEBUG
DEBUG = False

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config.yaml')
config = load_yaml(TRAIN_CONFIG_PATH)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# DATALOADER
NUM_WORKERS = config['DATALOADER']['num_workers']
CHECKPOINT_PATH = ''
PIN_MEMORY = config['DATALOADER']['pin_memory']

# TRAIN
EPOCHS = config['TRAIN']['num_epochs']
TRAIN_BATCH_SIZE = config['TRAIN']['batch_size']
MODEL = config['TRAIN']['model']
LEARNING_RATE = config['TRAIN']['learning_rate']
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']
OPTIMIZER = config['TRAIN']['optimizer']
SCHEDULER = config['TRAIN']['scheduler']
MOMENTUM = config['TRAIN']['momentum']
WEIGHT_DECAY = config['TRAIN']['weight_decay']
LOSS_FN = config['TRAIN']['loss_function']
METRIC_FN = config['TRAIN']['metric_function']
# VALIDATION
EVAL_BATCH_SIZE = config['VALIDATION']['batch_size']

# TRAIN SERIAL
KST = timezone(timedelta(hours=9))
TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
TRAIN_SERIAL = f'{TRAIN_TIMESTAMP}' if DEBUG is not True else 'DEBUG'

# PERFORMANCE RECORD
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'results', 'train', TRAIN_SERIAL)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    ENCODER = 'timm-tf_efficientnet_lite4'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['hair']
    ACTIVATION = 'sigmoid'

    # Set device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set train result directory
    make_directory(PERFORMANCE_RECORD_DIR)

    # Set system logger
    system_logger = get_logger(name='train', 
                               file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'train_log.log'))
    
    # Unet / PSPNet / DeepLabV3Plus
    if MODEL == 'unet':
        model = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    elif MODEL == 'pspnet':
        model = smp.PSPNet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    elif MODEL == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    elif  MODEL == 'pannet':
        model = smp.PAN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    elif  MODEL == 'fpn':
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    else:
        raise RuntimeError('Model name Error')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # ------------------------------------------------------
    # 데이터 로드

    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')


    def get_preprocessing(preprocessing_fn):
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
        return albu.Compose(_transform)


    train_dataset = Dataset_tu(
        mode = 'train',
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    val_dataset = Dataset_tu(
        mode = 'valid',
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    
    # Load dataset & dataloader
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)
    valid_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)


    # -------------------------------------------------------------
    # 모델 학습

    loss = smp.utils.losses.DiceLoss()

    # IOU / Fscore / Accuracy
    metrics = [
        smp.utils.metrics.IoU(threshold=iou_thres),
        smp.utils.metrics.Fscore(threshold=iou_thres),
        smp.utils.metrics.Accuracy(threshold=iou_thres),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=LEARNING_RATE),
    ])
    if CHECKPOINT_PATH != '':
        save_path = os.path.join(CHECKPOINT_PATH, MODEL)
        model = torch.load(os.path.join(save_path, f'best_model.pth'))
        print('load complete')
    else:
        save_path = './checkpoints/' + MODEL
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)


    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )


    
    # Load Model
    system_logger.info('===== Review Model Architecture =====')
    system_logger.info(f'{model} \n')

    # Set optimizer, scheduler, loss function, metric function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.0001, epochs=EPOCHS, steps_per_epoch=len(train_loader))
   
    #metric_fn = mean_squared_error

    # Set trainer
    trainer = CustomTrainer(model, DEVICE, loss, metrics, optimizer, scheduler, logger=system_logger)

    # Set earlystopper
    early_stopper = LossEarlyStopper(patience=EARLY_STOPPING_PATIENCE, verbose=True, logger=system_logger)

    # Set performance recorder
    key_column_value_list = [
        TRAIN_SERIAL,
        TRAIN_TIMESTAMP,
        MODEL,
        OPTIMIZER,
        LOSS_FN,
        METRIC_FN,
        EARLY_STOPPING_PATIENCE,
        TRAIN_BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        RANDOM_SEED]

    performance_recorder = PerformanceRecorder(column_name_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list,
                                               logger=system_logger,
                                               model=model,
                                               optimizer=optimizer,
                                               scheduler=scheduler)

    # Save config yaml file
    save_yaml(os.path.join(PERFORMANCE_RECORD_DIR, 'train_config.yaml'), config)

    # Train
    for epoch_index in tqdm(range(EPOCHS)):

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        # print('epoch : ',epoch)
        print('train_logs', train_logs)
        print('val_logs', valid_logs)

        val_loss = valid_logs['dice_loss']

        
        # Performance record - csv & save elapsed_time
        performance_recorder.add_row(epoch_index=epoch_index,
                                     train_loss=trainer.train_loss_mean,
                                     validation_loss=trainer.validation_loss_mean,
                                     train_score=trainer.train_score,
                                     validation_score=trainer.validation_score)

        # Performance record - plot
        performance_recorder.save_performance_plot(final_epoch=epoch_index)

        # early_stopping check
        early_stopper.check_early_stopping(loss=trainer.validation_loss_mean)
        
        if early_stopper.stop:
            break

        trainer.clear_history()

    # last model save
    performance_recorder.weight_path = os.path.join(PERFORMANCE_RECORD_DIR, 'last.pt')
    performance_recorder.save_weight()