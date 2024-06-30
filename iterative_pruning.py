import os
import torch
import argparse
import torch_pruning as tp
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import transforms, datasets
from models_training import ImageClassificationModel, TrainingTimeLogger, DEFAULT_PARAMS
from torchinfo import summary
from lightning.pytorch import seed_everything
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import NeptuneLogger

def parse_arguments():
    parser = argparse.ArgumentParser(description='Iterative pruning utilizing depgraph')
    parser.add_argument('--model_state_dict', type=str, required=True,          help='Path to model state dictionary')

    return parser.parse_args()

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

if __name__ == '__main__':
    total_cpu = os.cpu_count()

    # Seed for reproducibility
    seed_everything(42, workers=True)

    args = parse_arguments()

    # Hyperparameters
    DEFAULT_PARAMS.update({'pruning': True})

    # Prefix path for saving model checkpoint
    base_path = os.path.normpath(args.model_state_dict)
    parts = base_path.split('\\')
    index = parts.index('state_dictionaries')
    prefix_path = os.path.join(*parts[:index + 3])

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

    state_dict = torch.load(args.model_state_dict)

    model_name = state_dict.get('model_name', 'Unknown')

    state_dict.pop('model_name', None)

    model = ImageClassificationModel(train_dataset, val_dataset, test_dataset, model_name=model_name, num_classes=2)
    model.load_state_dict(state_dict)

    example_inputs = torch.rand(128, 3, 128, 128)
    
    # Importance criterion 
    imp = tp.importance.GroupNormImportance(p=2)

    # Ignore some layers that should not be pruned, e.g., the final classifier layer.
    if model_name == 'resnet18':
        ignored_layers = [model.model.fc]
    elif model_name == 'efficientnetv2s':
        ignored_layers = [model.model.classifier[1]]
    else:
        ignored_layers = [model.model.classifier[-1]]

    # Pruner initialization
    iterative_steps = 5
    pruner = tp.pruner.GroupNormPruner(
        model.model, 
        example_inputs, 
        global_pruning=False,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=0.5,
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

    for i in range(iterative_steps):
        pruner.step()

        macs, nparams = tp.utils.count_ops_and_params(model.model, example_inputs)
        print(model.model(example_inputs).shape)
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        )
        print(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/{prefix_path}/{model_name}/iteration_{i+1}',
            filename='{epoch}-{step}', 
            monitor='val_loss', 
            save_top_k=1, 
            mode='min')

        # Neptune Logger
        neptune_logger = NeptuneLogger(
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRiNTdmMS1mNDU0LTQxOTktYjBhYy02MDM4Y2ExOGRjOTAifQ==",  # Accessing the token from environment variable
            project="mzee/final-year-projects",  # replace with your Neptune project
            name=f'pruning_iteration_{i+1}',
            tags=["pytorch-lightning", "classification"]  # optional, for better organization
        )

        if i == 0:
            neptune_logger.log_hyperparams(DEFAULT_PARAMS)

        # Fine-tune the pruned model
        model.reset_optimizer_scheduler()
        model.train()
        trainer = pl.Trainer(fast_dev_run=False, max_epochs=5, logger=neptune_logger, callbacks=[training_time_logger, checkpoint_callback, lr_monitor, rich_progress_bar], precision='16-mixed' if model.use_amp else 32, accelerator='gpu' if torch.cuda.is_available() else 'cpu', deterministic=True)

        trainer.fit(model)

        neptune_logger.experiment.stop()


