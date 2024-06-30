import argparse
import torch
import os
import glob
import torch_pruning as tp
from models_training import ImageClassificationModel

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    return model

def model_benchmark(entire_model_dir=None, state_dict_dir=None, iterative_steps=None, model_state_dict=None, pruned=None):
    if pruned:
        for i in range(iterative_steps):
            iteration_model_dir = os.path.join(entire_model_dir, f'iteration_{i+1}')
            iteration_dict_dir = os.path.join(state_dict_dir, f'iteration_{i+1}')
            model_paths = glob.glob(os.path.join(iteration_model_dir, '*.pth'))
            state_dict_paths = glob.glob(os.path.join(iteration_dict_dir, '*.pth'))

            if model_paths and state_dict_paths:
                for idx, model_path in enumerate(model_paths):
                    print(f"Model found: {model_path}")
                    print(f"State dict found: {state_dict_paths[idx]}")
                    model = torch.load(model_path)
                    state_dict = torch.load(state_dict_paths[idx])
                    state_dict.pop('model_name', None)
                    model.load_state_dict(state_dict)
                    model.cuda()
                    device = torch.device('cuda:0')
                    example_inputs = torch.randn(1, 3, 128, 128, device='cuda')
                    # Test forward in eval mode
                    print("====== Forward (Inferece with torch.no_grad) ======")
                    with torch.no_grad():
                        laterncy_mu, latency_std= tp.utils.benchmark.measure_latency(model, example_inputs, repeat=300)
                        print('latency: {:.4f} +/- {:.4f} ms'.format(laterncy_mu, latency_std))

                        memory = tp.utils.benchmark.measure_memory(model, example_inputs, device=device)
                        print('memory: {:.4f} MB'.format(memory/ (1024)**2))

                        example_inputs_bs1 = torch.randn(1, 3, 128, 128, device='cuda')
                        fps = tp.utils.benchmark.measure_fps(model, example_inputs_bs1, repeat=300)
                        print('fps: {:.4f}'.format(fps))

                        example_inputs = torch.randn(128, 3, 128, 128, device='cuda')
                        throughput = tp.utils.benchmark.measure_throughput(model, example_inputs, repeat=300)
                        print('throughput (bs=128): {:.4f} images/s'.format(throughput))
    else:
        state_dict = torch.load(model_state_dict)
        model_name = state_dict.get('model_name', 'Unknown')
        state_dict.pop('model_name', None)

        if model_name == 'efficientnetv2s':
            model = ImageClassificationModel(model_name='efficientnetv2s', num_classes=2)
            model.load_state_dict(state_dict)
        elif model_name == 'resnet18':
            model = ImageClassificationModel(model_name='resnet18', num_classes=2)
            model.load_state_dict(state_dict)
        else:
            model = ImageClassificationModel(model_name='mobilenetv3s', num_classes=2)
            model.load_state_dict(state_dict)

        model.cuda()
        print(f'Loaded state dictionary for model: {model_name}')
        device = torch.device('cuda:0')
        example_inputs = torch.randn(1, 3, 128, 128, device='cuda')
        # Test forward in eval mode
        print("====== Forward (Inferece with torch.no_grad) ======")
        with torch.no_grad():
            laterncy_mu, latency_std= tp.utils.benchmark.measure_latency(model, example_inputs, repeat=300)
            print('latency: {:.4f} +/- {:.4f} ms'.format(laterncy_mu, latency_std))

            memory = tp.utils.benchmark.measure_memory(model, example_inputs, device=device)
            print('memory: {:.4f} MB'.format(memory/ (1024)**2))

            example_inputs_bs1 = torch.randn(1, 3, 128, 128, device='cuda')
            fps = tp.utils.benchmark.measure_fps(model, example_inputs_bs1, repeat=300)
            print('fps: {:.4f}'.format(fps))

            example_inputs = torch.randn(128, 3, 128, 128, device='cuda')
            throughput = tp.utils.benchmark.measure_throughput(model, example_inputs, repeat=300)
            print('throughput (bs=128): {:.4f} images/s'.format(throughput))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Benchmark model\'s performance')
    parser.add_argument('--entire_model_dir', type=str, help='Path to checkpoint directory')
    parser.add_argument('--state_dict_dir', type=str, help='Path to state dict directory')
    parser.add_argument('--iterative_steps', type=int, help='Iterative steps while doing model pruning')
    parser.add_argument('--model_state_dict', type=str, help='Path to model state dict')
    parser.add_argument('--pruned', action='store_true', help='Pruned mode (Default: False)')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    model_benchmark(args.entire_model_dir, args.state_dict_dir, args.iterative_steps, args.model_state_dict, args.pruned)