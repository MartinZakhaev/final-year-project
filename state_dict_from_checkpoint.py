import torch
import argparse
import os
import glob
import torch_pruning as tp
from models_training import ImageClassificationModel

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    return model

def save_state_dict(checkpoint_dir=None, model_state_dict_path=None, model_checkpoint_path=None, pruned=None):
    if pruned:
        print(f"Loading state_dict from: {model_state_dict_path}")
        state_dict = torch.load(model_state_dict_path)

        model_name = state_dict.get('model_name', 'Unknown')

        state_dict.pop('model_name', None)

        model = ImageClassificationModel(model_name=model_name, num_classes=2)
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
            iteration_dir = os.path.join(checkpoint_dir, f'iteration_{i+1}')
            checkpoint_paths = glob.glob(os.path.join(iteration_dir, '*.ckpt'))
            if checkpoint_paths:
                for checkpoint_path in checkpoint_paths:
                    print(f"Checkpoint iteration {i + 1} found: {checkpoint_path}")
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

                    checkpoint_path = os.path.normpath(checkpoint_path)
                    to_be_saved = load_checkpoint(model, checkpoint_path)
                    state_dict = to_be_saved.state_dict()

                    # Include the model name in the state dictionary
                    state_dict['model_name'] = model.hparams.model_name

                    # Extract the necessary parts from the checkpoint path
                    base_name = os.path.basename(checkpoint_path)
                    dir_name = os.path.dirname(checkpoint_path)
                    save_state_dict_dir = os.path.join('state_dictionaries', dir_name)
                    save_model_dir = os.path.join('entire_models', dir_name)

                    # Ensure the save directory exists
                    os.makedirs(save_state_dict_dir, exist_ok=True)
                    os.makedirs(save_model_dir, exist_ok=True)

                    # Create the save path for the state dictionary
                    save_dict_path = os.path.join(save_state_dict_dir, base_name.replace('.ckpt', '.pth'))
                    save_model_path = os.path.join(save_model_dir, base_name.replace('.ckpt', '.pth'))

                    # Save the state dictionary
                    torch.save(state_dict, save_dict_path)
                    print(f'Saved state dictionary at {save_dict_path}')

                    # Save the entire model
                    torch.save(to_be_saved, save_model_path)
                    print(f'Saved model at {save_model_path}')
            else:
                print(f"Checkpoint for iteration {i} not found in {iteration_dir}")
    else:
        print(f"Loading checkpoint from: {model_checkpoint_path}")
        checkpoint_path = os.path.normpath(model_checkpoint_path)
        model = ImageClassificationModel.load_from_checkpoint(checkpoint_path)
        state_dict = model.state_dict()

        # Include the model name in the state dictionary
        state_dict['model_name'] = model.hparams.model_name

        # Extract the necessary parts from the checkpoint path
        base_name = os.path.basename(checkpoint_path)
        dir_name = os.path.dirname(checkpoint_path)
        save_dir = os.path.join('state_dictionaries', dir_name)

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Create the save path for the state dictionary
        save_path = os.path.join(save_dir, base_name.replace('.ckpt', '.pth'))

        # Save the state dictionary
        torch.save(state_dict, save_path)
        print(f'Saved state dictionary at {save_path}')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert model\'s checkpoint to state dictionary')
    parser.add_argument('--checkpoint_dir', type=str, help='Path to checkpoint directory')
    parser.add_argument('--model_state_dict', type=str, help='Path to model state dictionary')
    parser.add_argument('--model_checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--pruned', action='store_true', help='Pruned mode (Default: False)')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print(f"Arguments received: checkpoint_dir={args.checkpoint_dir}, model_state_dict={args.model_state_dict}, model_checkpoint={args.model_checkpoint}, pruned={args.pruned}")

    save_state_dict(args.checkpoint_dir, args.model_state_dict, args.model_checkpoint, args.pruned)
