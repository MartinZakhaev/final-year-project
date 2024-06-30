import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
import neptune
import torchmetrics.classification
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

total_cpu = os.cpu_count()

# Hyperparameters
DEFAULT_PARAMS = {
    'model_name': 'resnet18',
    'img_size': 128,
    'num_classes': 2,
    'batch_size': 128,
    'num_epochs': 5,
    'num_workers': int(total_cpu / 2),
    'learning_rate': 1e-2,
    'momentum': 0.9,
    'gamma': 0.9,
    'weight_decay': 1e-4,
    'criterion': 'cross entropy loss',
    'optimizer': 'sgd',
    'dev': False,
}

class ResNetModel(nn.Module):
    def __init__(self, num_classes, model_name='resnet18'):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes, model_name='efficientnetv2s'):
        super().__init__()
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

class MobileNetV3SmallModel(nn.Module):
    def __init__(self, num_classes, model_name='mobilenetv3s'):
        super().__init__()
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

class ImageClassificationModel(pl.LightningModule):
    def __init__(self, train_dataset = None, val_dataset = None, test_dataset = None, model_name='resnet18', learning_rate=1e-2, momentum=0.9, gamma=0.9, weight_decay=1e-4, num_classes=2, use_amp=True):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(model_name, num_classes).model
        self.criterion = nn.CrossEntropyLoss()
        self.use_amp = use_amp
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.train_loss(loss)
        self.train_accuracy(outputs, labels)
        self.train_precision(outputs, labels)
        self.train_recall(outputs, labels)
        self.train_f1(outputs, labels)
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision)
        self.log('train_recall', self.train_recall)
        self.log('train_f1', self.train_f1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.val_loss(loss)
        self.val_accuracy(outputs, labels)
        self.val_precision(outputs, labels)
        self.val_recall(outputs, labels)
        self.val_f1(outputs, labels)
        self.log('val_loss', self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision)
        self.log('val_recall', self.val_recall)
        self.log('val_f1', self.val_f1)
        return {'val_loss': loss, 'val_accuracy': self.val_accuracy}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
    
    def reset_optimizer_scheduler(self):
        self.optimizers, self.schedulers = self.configure_optimizers()
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=DEFAULT_PARAMS['batch_size'], shuffle=True, num_workers=DEFAULT_PARAMS['num_workers'], persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=DEFAULT_PARAMS['batch_size'], shuffle=False, num_workers=DEFAULT_PARAMS['num_workers'], persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=DEFAULT_PARAMS['batch_size'], shuffle=False, num_workers=DEFAULT_PARAMS['num_workers'], persistent_workers=True, pin_memory=True)
    
class TrainingTimeLogger(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.epoch_times = []

    def on_train_epoch_end(self, trainer, pl_module):
        current_time = time.time()
        epoch_time = current_time - self.start_time
        self.epoch_times.append(epoch_time)
        self.start_time = current_time
        trainer.logger.experiment["epoch_time"].log(epoch_time)

    def on_train_end(self, trainer, pl_module):
        total_time = sum(self.epoch_times)
        trainer.logger.experiment["total_training_time"].log(total_time)

def get_model(model_name, num_classes):
    if model_name == 'resnet18':
        return ResNetModel(num_classes, model_name='resnet18')
    elif model_name == 'efficientnetv2s':
        return EfficientNetModel(num_classes, model_name='efficientnetv2s')
    elif model_name == 'mobilenetv3s':
        return MobileNetV3SmallModel(num_classes, model_name='mobilenetv3s')
    else:
        raise NotImplementedError(f"Model '{model_name}' not implemented or supported.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train an image classification model')
    parser.add_argument('--model_name', type=str, default=DEFAULT_PARAMS['model_name'], choices=['resnet18', 'efficientnetv2s', 'mobilenetv3s'],
                        help='Name of the model architecture to use (default: resnet18)')
    parser.add_argument('--num_classes', type=int, default=DEFAULT_PARAMS['num_classes'],
                        help='Number of classes in the dataset (default: 2)')
    parser.add_argument('--img_size', type=int, default=DEFAULT_PARAMS['img_size'],
                        help='Input image size (default: 128)')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_PARAMS['batch_size'],
                        help='Batch size for training and validation (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_PARAMS['num_epochs'],
                        help='Number of epochs to train (default: 5)')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_PARAMS['num_workers'],
                        help='Number of workers (cpu core) for data loader (default: half total cpu cores available)')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_PARAMS['learning_rate'],
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=DEFAULT_PARAMS['momentum'],
                        help='Momentum for SGD optimizer (default: 0.9)')
    parser.add_argument('--gamma', type=float, default=DEFAULT_PARAMS['gamma'],
                        help='Gamma parameter for learning rate scheduler (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_PARAMS['weight_decay'],
                        help='Weight decay (L2 penalty) (default: 0.0001)')
    parser.add_argument('--dev', type=bool, default=DEFAULT_PARAMS['dev'],
                        help='Running code in a fast dev mode (default: False)')
    return parser.parse_args()

if __name__ == "__main__":
    # Seed for reproducibility
    seed_everything(42, workers=True)

    args = parse_arguments()

    PARAMS = {
        'model_name': args.model_name,
        'img_size': args.img_size,
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'num_workers': args.num_workers,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'gamma': args.gamma,
        'weight_decay': args.weight_decay,
        'criterion': DEFAULT_PARAMS['criterion'],
        'optimizer': DEFAULT_PARAMS['optimizer'],
        'dev': args.dev,
    }

    # Data transformations
    train_transforms = transforms.Compose([
        transforms.Resize((DEFAULT_PARAMS['img_size'], DEFAULT_PARAMS['img_size'])),
        transforms.RandomRotation(25),
        transforms.RandomResizedCrop(DEFAULT_PARAMS['img_size'], scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((DEFAULT_PARAMS['img_size'], DEFAULT_PARAMS['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and Dataloaders
    train_dataset = datasets.ImageFolder("C:/FINAL YEAR PLAYGROUND/dataset bank/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/train", transform=train_transforms)
    val_dataset = datasets.ImageFolder("C:/FINAL YEAR PLAYGROUND/dataset bank/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/valid", transform=val_transforms)
    test_dataset = datasets.ImageFolder("C:/FINAL YEAR PLAYGROUND/dataset bank/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/test", transform=val_transforms)

    # Neptune Logger
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRiNTdmMS1mNDU0LTQxOTktYjBhYy02MDM4Y2ExOGRjOTAifQ==",  # Accessing the token from environment variable
        project="mzee/final-year-projects",  # replace with your Neptune project
        tags=["pytorch-lightning", "classification"]  # optional, for better organization
    )

    neptune_logger.log_hyperparams(PARAMS)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    rich_progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        )
    )
    training_time_logger = TrainingTimeLogger()

    # Training
    model = ImageClassificationModel(train_dataset, val_dataset, test_dataset, model_name=PARAMS['model_name'], num_classes=PARAMS['num_classes'], learning_rate=PARAMS['learning_rate'], momentum=PARAMS['momentum'], gamma=PARAMS['gamma'], weight_decay=PARAMS['weight_decay'])

    trainer = pl.Trainer(fast_dev_run=PARAMS['dev'], max_epochs=PARAMS['num_epochs'], logger=neptune_logger, callbacks=[training_time_logger, checkpoint_callback, lr_monitor, rich_progress_bar], precision='16-mixed' if model.use_amp else 32, accelerator='gpu' if torch.cuda.is_available() else 'cpu', deterministic=True)

    trainer.fit(model)

    neptune_logger.log_model_summary(model=model, max_depth=-1)