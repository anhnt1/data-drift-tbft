# --- START OF FILE drift_exp.py ---

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, random_split, ConcatDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
import matplotlib
import argparse
import os
import copy
from tqdm import tqdm
from itertools import chain
import time
import csv
import sys # For exiting on critical errors

try:
    matplotlib.use('Agg')
    print("Using Matplotlib backend: Agg (non-interactive)")
except ImportError:
    print("Warning: Agg backend not found. Plots might not save correctly without a display.")

import matplotlib.pyplot as plt

# Base SEED for initial setup (like train/val split if done once)
# Run-specific seeds will be derived from this
BASE_SEED = 42
# Set initial seeds for consistency in setup before the run loop
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)
    # These should be set globally for deterministic operations where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

DATASET_CONFIGS = {
    'cifar10': {
        'data_root': './cifar10_drift_sim',
        'dataset_class': torchvision.datasets.CIFAR10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
        'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        'num_classes': 10,
        'channels': 3,
        'input_size': 32,
         'base_transform': lambda mean, std: transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'augment_transform': lambda mean, std: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'input_drift_transforms': lambda mean, std: transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1), # Relies on torch seeding
            transforms.Normalize(mean, std)
        ]),
        'feature_drift_source_indices': {0, 1, 2, 3, 4}, # Mixed split
        'feature_drift_target_indices': {5, 6, 7, 8, 9}, # Mixed split
        'output_drift_map': {3: 5, 5: 3}, # cat <-> dog
        'feature_drift_title': "Feature Drift (Mixed Easy/Hard Split)",
        'output_drift_title': "Output Drift (cat <-> dog swapped)"
    },
    'svhn': {
        'data_root': './svhn_drift_sim',
        'dataset_class': torchvision.datasets.SVHN,
        'mean': (0.4377, 0.4438, 0.4728),
        'std': (0.1980, 0.2010, 0.1970),
        'classes': tuple(str(i) for i in range(10)),
        'num_classes': 10,
        'channels': 3,
        'input_size': 32,
         'base_transform': lambda mean, std: transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'augment_transform': lambda mean, std: transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)), # Relies on torch seeding
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'input_drift_transforms': lambda mean, std: transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Relies on torch seeding
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1), # Relies on torch seeding
            transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)), # Relies on torch seeding
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1), # Relies on torch seeding
            transforms.Normalize(mean, std)
        ]),
        'feature_drift_source_indices': {0, 1, 2, 3, 4},
        'feature_drift_target_indices': {5, 6, 7, 8, 9},
        'output_drift_map': {1: 7, 7: 1},
        'feature_drift_title': "Feature Drift (Source: Digits 0-4, Target: Digits 5-9)",
        'output_drift_title': "Output Drift (Digits 1 <-> 7 swapped)"
    }
}

class SVHNWrapper(Dataset):
    def __init__(self, svhn_dataset):
        self.svhn_dataset = svhn_dataset
        if hasattr(svhn_dataset, 'labels'):
            raw_labels = svhn_dataset.labels
            self._mapped_labels = torch.tensor(raw_labels) % 10
        else:
            print("Warning: Could not find 'labels' attribute in SVHN dataset. Label mapping might fail.")
            self._mapped_labels = torch.tensor([self.svhn_dataset[i][1] % 10 for i in range(len(self.svhn_dataset))])

    def __len__(self):
        return len(self.svhn_dataset)

    def __getitem__(self, idx):
        image, _ = self.svhn_dataset[idx]
        label = self._mapped_labels[idx]
        return image, label.long()

    @property
    def labels(self):
        return self._mapped_labels

    @property
    def targets(self):
        if not hasattr(self, '_cached_targets_list'):
             self._cached_targets_list = self._mapped_labels.tolist()
        return self._cached_targets_list

    def __getattr__(self, name):
        return getattr(self.svhn_dataset, name)

def get_base_data(config, dataset_name, train=True, augment=False):
    if train and augment:
        transform = config['augment_transform'](config['mean'], config['std'])
    else:
        transform = config['base_transform'](config['mean'], config['std'])

    if dataset_name == 'svhn':
        split = 'train' if train else 'test'
        raw_svhn_dataset = config['dataset_class'](
            root=config['data_root'],
            split=split,
            download=True,
            transform=transform
        )
        dataset = SVHNWrapper(raw_svhn_dataset)
    elif dataset_name == 'cifar10':
        dataset = config['dataset_class'](
            root=config['data_root'],
            train=train,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset name in get_base_data: {dataset_name}")
    return dataset

def unnormalize(tensor, mean, std):
    tensor = tensor.clone()
    if not isinstance(mean, (list, tuple)): mean = (mean,)
    if not isinstance(std, (list, tuple)): std = (std,)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_drift(source_loader, target_loader, title, config, save_path, num_samples=5):
    print(f"\n--- Visualizing {title} ---")
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    try:
        source_images, source_labels = next(source_iter)
        target_images, target_labels = next(target_iter)
    except StopIteration:
        print(f"Warning: Could not get enough samples to visualize for {title}.")
        return

    plt.figure(figsize=(12, 6))
    dataset_obj = source_loader.dataset
    while isinstance(dataset_obj, (Subset, SVHNWrapper, LabelShiftingDataset)):
        if hasattr(dataset_obj, 'dataset'): dataset_obj = dataset_obj.dataset
        elif hasattr(dataset_obj, 'base_dataset'): dataset_obj = dataset_obj.base_dataset
        elif hasattr(dataset_obj, 'svhn_dataset'): dataset_obj = dataset_obj.svhn_dataset
        else: break
    dataset_name_viz = type(dataset_obj).__name__
    plt.suptitle(f"{dataset_name_viz}: {title}", fontsize=16)

    actual_samples = min(num_samples, len(source_images), len(target_images))
    if actual_samples < num_samples:
         print(f"Warning: Displaying only {actual_samples} samples due to batch size.")
    if actual_samples == 0:
        print("Error: No samples to display.")
        plt.close()
        return

    mean, std = config['mean'], config['std']
    class_names = config.get('classes', None)
    channels = config['channels']

    for i in range(actual_samples):
        plt.subplot(2, actual_samples, i + 1)
        img_display_source = unnormalize(source_images[i], mean, std)
        label_idx_source = source_labels[i].item()
        source_label_str = class_names[label_idx_source] if class_names and 0 <= label_idx_source < len(class_names) else f"Idx {label_idx_source}"
        title_source = f"Source: {source_label_str}"
        if channels == 1: plt.imshow(img_display_source.squeeze().numpy(), cmap='gray')
        else: plt.imshow(img_display_source.permute(1, 2, 0).numpy().clip(0, 1))
        plt.title(title_source, fontsize=8)
        plt.axis('off')

        plt.subplot(2, actual_samples, actual_samples + i + 1)
        img_display_target = unnormalize(target_images[i], mean, std)
        label_idx_target = target_labels[i].item()
        target_label_str = class_names[label_idx_target] if class_names and 0 <= label_idx_target < len(class_names) else f"Idx {label_idx_target}"
        title_target = f"Target: {target_label_str}"
        if channels == 1: plt.imshow(img_display_target.squeeze().numpy(), cmap='gray')
        else: plt.imshow(img_display_target.permute(1, 2, 0).numpy().clip(0, 1))
        plt.title(title_target, fontsize=8)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()

def train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'Loss': running_loss / (pbar.n + 1e-6), 'Acc': 100. * correct / (total + 1e-6)})
    train_loss = running_loss / (len(loader) + 1e-6)
    train_acc = 100. * correct / (total + 1e-6)
    return train_loss, train_acc

def test_epoch(loader, model, criterion, device, desc='Testing'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'Loss': running_loss / (pbar.n + 1e-6), 'Acc': 100. * correct / (total + 1e-6)})
    if not total: return 0.0, 0.0
    test_loss = running_loss / (len(loader) + 1e-6)
    test_acc = 100. * correct / (total + 1e-6)
    return test_loss, test_acc

def add_noise_to_layer(layer, std_dev=0.01):
    if layer is None:
        print("Warning: Layer not found for noise addition.")
        return
    with torch.no_grad():
        for param in layer.parameters():
            noise = torch.randn_like(param) * std_dev # Uses torch seed
            param.add_(noise)

def adapt_model(model_name, num_classes, channels, pretrained=False):
    weights = None
    if pretrained:
        if model_name == 'resnet18': weights = models.ResNet18_Weights.DEFAULT
        elif model_name == 'densenet121': weights = models.DenseNet121_Weights.DEFAULT
        elif model_name == 'resnext50': weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        else: print(f"Warning: Pretrained weights not specified for {model_name} in adapt_model.")

    if model_name == 'resnet18': model = models.resnet18(weights=weights)
    elif model_name == 'densenet121': model = models.densenet121(weights=weights)
    elif model_name == 'resnext50': model = models.resnext50_32x4d(weights=weights)
    else: raise ValueError(f"Unsupported model: {model_name}")

    if channels != 3:
         print(f"Warning: Adapting input layer for {channels} channels. Check if necessary.")
         if hasattr(model, 'conv1'):
             original_conv1 = model.conv1
             model.conv1 = nn.Conv2d(channels, original_conv1.out_channels,
                                     kernel_size=original_conv1.kernel_size, stride=original_conv1.stride,
                                     padding=original_conv1.padding, bias=original_conv1.bias is not None)
             if pretrained and original_conv1.weight.data.shape[1] == 3:
                 if channels == 1: model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
                 else: model.conv1.weight.data = original_conv1.weight.data[:, :channels, :, :]
         elif hasattr(model, 'features') and hasattr(model.features, 'conv0'):
             original_conv0 = model.features.conv0
             model.features.conv0 = nn.Conv2d(channels, original_conv0.out_channels,
                                              kernel_size=original_conv0.kernel_size, stride=original_conv0.stride,
                                              padding=original_conv0.padding, bias=original_conv0.bias is not None)
             if pretrained and original_conv0.weight.data.shape[1] == 3:
                 if channels == 1: model.features.conv0.weight.data = original_conv0.weight.data.mean(dim=1, keepdim=True)
                 else: model.features.conv0.weight.data = original_conv0.weight.data[:, :channels, :, :]
         else: print(f"Warning: Could not automatically adapt input layer for {model_name} with {channels} channels.")

    if hasattr(model, 'fc'):
        num_ftrs = model.fc.in_features
        if model.fc.out_features != num_classes:
            print(f"Replacing final FC layer: {model.fc.out_features} -> {num_classes} classes.")
            model.fc = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            num_ftrs = model.classifier.in_features
            if model.classifier.out_features != num_classes:
                 print(f"Replacing final Classifier layer: {model.classifier.out_features} -> {num_classes} classes.")
                 model.classifier = nn.Linear(num_ftrs, num_classes)
        else: print(f"Warning: Classifier adaptation for {model_name} might need specific handling.")
    else: print(f"Warning: Could not automatically adapt output layer for {model_name}.")
    return model

def get_layer_groups(model, model_name):
    groups = {}
    if model_name == 'resnet18':
        groups['conv1_layer1'] = chain(model.conv1.parameters(), model.layer1.parameters())
        groups['layer2'] = model.layer2.parameters()
        groups['layer3'] = model.layer3.parameters()
        groups['layer4'] = model.layer4.parameters()
        groups['fc'] = model.fc.parameters()
    elif model_name == 'densenet121':
        groups['conv0_dense1'] = chain(model.features.conv0.parameters(), model.features.denseblock1.parameters())
        groups['dense2_trans2'] = chain(model.features.denseblock2.parameters(), model.features.transition2.parameters())
        groups['dense3_trans3'] = chain(model.features.denseblock3.parameters(), model.features.transition3.parameters())
        groups['dense4_norm5'] = chain(model.features.denseblock4.parameters(), model.features.norm5.parameters())
        groups['classifier'] = model.classifier.parameters()
    elif model_name == 'resnext50':
        groups['conv1_layer1'] = chain(model.conv1.parameters(), model.layer1.parameters())
        groups['layer2'] = model.layer2.parameters()
        groups['layer3'] = model.layer3.parameters()
        groups['layer4'] = model.layer4.parameters()
        groups['fc'] = model.fc.parameters()
    else:
         raise ValueError(f"Layer groups not defined for model: {model_name}")
    groups['all'] = model.parameters()
    return groups

def get_final_classifier_layer(model, model_name):
    if model_name in ['resnet18', 'resnext50']:
        return model.fc
    elif model_name == 'densenet121':
        if isinstance(model.classifier, nn.Linear): return model.classifier
        else: return None
    else:
        return None

class LabelShiftingDataset(Dataset):
    def __init__(self, base_dataset, shift_map):
        self.base_dataset = base_dataset
        self.shift_map = shift_map
        try:
            if hasattr(base_dataset, 'labels') and base_dataset.labels is not None:
                 self.original_targets = torch.tensor(base_dataset.labels) if not isinstance(base_dataset.labels, torch.Tensor) else base_dataset.labels.clone()
            elif hasattr(base_dataset, 'targets') and base_dataset.targets is not None:
                 self.original_targets = torch.tensor(base_dataset.targets) if not isinstance(base_dataset.targets, torch.Tensor) else base_dataset.targets.clone()
            elif isinstance(base_dataset, Subset):
                 underlying_dataset = base_dataset.dataset
                 if hasattr(underlying_dataset, 'labels') and underlying_dataset.labels is not None:
                      all_labels = torch.tensor(underlying_dataset.labels) if not isinstance(underlying_dataset.labels, torch.Tensor) else underlying_dataset.labels
                      self.original_targets = all_labels[base_dataset.indices]
                 elif hasattr(underlying_dataset, 'targets') and underlying_dataset.targets is not None:
                      all_targets = torch.tensor(underlying_dataset.targets) if not isinstance(underlying_dataset.targets, torch.Tensor) else underlying_dataset.targets
                      self.original_targets = all_targets[base_dataset.indices]
                 else:
                      print("Warning: Could not access labels easily for Subset in LabelShiftingDataset. Iterating (slow).")
                      self.original_targets = torch.tensor([self.base_dataset[i][1] for i in range(len(self.base_dataset))])
            else:
                print("Warning: Could not directly access targets/labels in LabelShiftingDataset. Iterating (might be slow).")
                self.original_targets = torch.tensor([self.base_dataset[i][1] for i in range(len(self.base_dataset))])
        except Exception as e:
             print(f"Error accessing targets/labels in LabelShiftingDataset: {e}. Falling back to iteration.")
             self.original_targets = torch.tensor([self.base_dataset[i][1] for i in range(len(self.base_dataset))])
        self.shifted_targets = self._create_shifted_labels()

    def _create_shifted_labels(self):
        shifted_targets = self.original_targets.clone()
        for original_label, target_label in self.shift_map.items():
            mask = self.original_targets == original_label
            shifted_targets[mask] = target_label
        return shifted_targets

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        image = item[0]
        shifted_label = self.shifted_targets[idx]
        return image, shifted_label.long()

    @property
    def labels(self):
        return self.shifted_targets

    @property
    def targets(self):
        if not hasattr(self, '_cached_shifted_targets_list'):
             self._cached_shifted_targets_list = self.shifted_targets.tolist()
        return self._cached_shifted_targets_list

def simulate_input_drift(base_dataset, config):
    # This simulation uses transforms that depend on the torch random state
    print(f"Simulating Input-Level Drift...")
    source_dataset = base_dataset
    target_transform = config['input_drift_transforms'](config['mean'], config['std'])
    if isinstance(base_dataset, Subset):
        original_dataset = base_dataset.dataset
        if isinstance(original_dataset, SVHNWrapper):
            underlying_svhn = original_dataset.svhn_dataset
            original_class = type(underlying_svhn)
            original_root = getattr(underlying_svhn, 'root', None)
            original_split = getattr(underlying_svhn, 'split', None)
            target_base_raw = original_class(
                 root=original_root, split=original_split,
                 download=False, transform=target_transform
            )
            target_base_dataset = SVHNWrapper(target_base_raw)
        elif isinstance(original_dataset, torchvision.datasets.CIFAR10):
            original_class = type(original_dataset)
            original_root = getattr(original_dataset, 'root', None)
            original_train_flag = getattr(original_dataset, 'train', None)
            target_base_dataset = original_class(
                 root=original_root, train=original_train_flag,
                 download=False, transform=target_transform
            )
        else:
             raise RuntimeError("Cannot recreate base dataset for input drift simulation on this Subset type.")
        target_dataset = Subset(target_base_dataset, base_dataset.indices)
    else:
        original_class = type(base_dataset)
        if isinstance(base_dataset, SVHNWrapper):
             underlying_svhn = base_dataset.svhn_dataset
             original_root = getattr(underlying_svhn, 'root', None)
             original_split = getattr(underlying_svhn, 'split', None)
             target_base_raw = type(underlying_svhn)(
                 root=original_root, split=original_split,
                 download=False, transform=target_transform
             )
             target_dataset = SVHNWrapper(target_base_raw)
        elif isinstance(base_dataset, torchvision.datasets.CIFAR10):
             original_root = getattr(base_dataset, 'root', None)
             original_train_flag = getattr(base_dataset, 'train', None)
             target_dataset = original_class(
                 root=original_root, train=original_train_flag,
                 download=False, transform=target_transform
             )
        else:
             raise RuntimeError("Cannot recreate base dataset for input drift simulation.")
    return source_dataset, target_dataset

def simulate_feature_drift(base_dataset, config):
    # This simulation is deterministic given the base_dataset and config
    source_class_indices = config['feature_drift_source_indices']
    target_class_indices = config['feature_drift_target_indices']
    print(f"Simulating Feature-Level Drift...")
    current_dataset = base_dataset
    original_indices = list(range(len(current_dataset)))
    while isinstance(current_dataset, (Subset, SVHNWrapper)):
        if isinstance(current_dataset, Subset):
            original_indices = current_dataset.indices
            current_dataset = current_dataset.dataset
        elif isinstance(current_dataset, SVHNWrapper):
            current_dataset = current_dataset.svhn_dataset
        else: break
    original_dataset = current_dataset
    try:
        if hasattr(original_dataset, 'targets'):
            all_labels = np.array(original_dataset.targets)
        elif hasattr(original_dataset, 'labels'):
             all_labels = np.array(original_dataset.labels) % 10
        else:
            print("Warning: .targets/labels not found in original dataset for feature drift, iterating (slow).")
            all_labels = np.array([original_dataset[i][1] for i in range(len(original_dataset))])
            if type(original_dataset) is torchvision.datasets.SVHN: all_labels = all_labels % 10
    except Exception as e:
        print(f"Error accessing labels for feature drift: {e}. Iterating.")
        all_labels = np.array([original_dataset[i][1] for i in range(len(original_dataset))])
        if type(original_dataset) is torchvision.datasets.SVHN: all_labels = all_labels % 10

    if isinstance(base_dataset, Subset):
        base_indices = base_dataset.indices
        labels_in_base = all_labels[base_indices]
        source_final_indices = [base_indices[i] for i, label in enumerate(labels_in_base) if label in source_class_indices]
        target_final_indices = [base_indices[i] for i, label in enumerate(labels_in_base) if label in target_class_indices]
    else:
        source_final_indices = [i for i, label in enumerate(all_labels) if label in source_class_indices]
        target_final_indices = [i for i, label in enumerate(all_labels) if label in target_class_indices]

    if not source_final_indices: print("Warning: No samples found for source domain in feature drift.")
    if not target_final_indices: print("Warning: No samples found for target domain in feature drift.")

    dataset_to_subset = base_dataset.dataset if isinstance(base_dataset, Subset) else base_dataset
    source_subset = Subset(dataset_to_subset, source_final_indices)
    target_subset = Subset(dataset_to_subset, target_final_indices)
    return source_subset, target_subset

def simulate_output_drift(base_dataset, config):
    # This simulation is deterministic given the base_dataset and config
    label_shift_map = config['output_drift_map']
    print(f"Simulating Output-Level Drift...")
    source_dataset = base_dataset
    target_dataset_shifted = LabelShiftingDataset(base_dataset, label_shift_map)
    return source_dataset, target_dataset_shifted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train, Simulate Drift, and Evaluate/Adapt Models')
    parser.add_argument('--datasets', nargs='+', default=['cifar10'], choices=['cifar10', 'svhn'], help='Datasets to use')
    parser.add_argument('--models', nargs='+', default=['resnet18'], choices=['resnet18', 'densenet121', 'resnext50'], help='Models to evaluate')
    parser.add_argument('--ft_ratios', nargs='+', type=float, default=[0.1], help='List of ratios of drifted data to use for fine-tuning')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of independent runs for statistical analysis')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation')
    parser.add_argument('--train_epochs', type=int, default=10, help='Epochs for baseline training')
    parser.add_argument('--ft_epochs', type=int, default=5, help='Max epochs for fine-tuning')
    parser.add_argument('--lr_base', type=float, default=0.01, help='Learning rate for baseline training')
    parser.add_argument('--lr_ft', type=float, default=0.001, help='Learning rate for fine-tuning')
    parser.add_argument('--early_stop_patience', type=int, default=3, help='Patience for early stopping during fine-tuning')
    parser.add_argument('--results_dir', type=str, default='./drift_results', help='Directory to save models, plots, and results CSV')
    parser.add_argument('--skip_train', action='store_true', help='Skip baseline training and attempt to load existing models')
    parser.add_argument('--skip_viz', action='store_true', help='Skip generating and saving drift visualization plots')
    parser.add_argument('--use_pretrained', action='store_true', help='Use ImageNet pretrained weights for model initialization')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    all_run_results = [] # Global list to store results from all runs/configs
    NUM_WORKERS_REPRODUCIBLE = 0 # Keep workers at 0 for deterministic data loading within runs

    # --- Dataset Loop ---
    for dataset_name in args.datasets:
        print(f"\n{'#'*25} Processing Dataset: {dataset_name.upper()} {'#'*25}")
        active_config = DATASET_CONFIGS[dataset_name]

        print(f"\n--- Loading Clean {dataset_name.upper()} Data ---")
        # Load data once per dataset
        clean_train_dataset_aug = get_base_data(active_config, dataset_name, train=True, augment=True)
        clean_train_dataset_noaug = get_base_data(active_config, dataset_name, train=True, augment=False)
        clean_test_dataset = get_base_data(active_config, dataset_name, train=False, augment=False)

        # Create train/val split once per dataset using BASE_SEED for consistency
        val_split_ratio = 0.1
        num_train = len(clean_train_dataset_noaug)
        indices = list(range(num_train))
        # Use a temporary RNG state for shuffling based on BASE_SEED if needed, or rely on initial np.random state
        # np.random.seed(BASE_SEED) # Resetting here might affect subsequent runs if not careful
        np.random.shuffle(indices) # Relies on initial BASE_SEED np state
        split = int(np.floor(val_split_ratio * num_train))
        train_idx, val_idx = indices[split:], indices[:split]
        clean_val_dataset = Subset(clean_train_dataset_noaug, val_idx)
        clean_train_subset_for_aug = Subset(clean_train_dataset_aug, train_idx)

        # Create base DataLoaders once per dataset
        base_train_loader = DataLoader(clean_train_subset_for_aug, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS_REPRODUCIBLE, pin_memory=True)
        base_val_loader = DataLoader(clean_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS_REPRODUCIBLE, pin_memory=True)
        base_test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS_REPRODUCIBLE, pin_memory=True)
        print(f"Train samples: {len(clean_train_subset_for_aug)}, Val samples: {len(clean_val_dataset)}, Test samples: {len(clean_test_dataset)}")

        # --- Model Loop ---
        for model_name in args.models:
            print(f"\n{'='*20} Processing Model: {model_name} for Dataset: {dataset_name} {'='*20}")
            model_save_path = os.path.join(args.results_dir, f"{dataset_name}_{model_name}_baseline.pth")
            # Initialize model architecture - this is cheap
            model = adapt_model(model_name, active_config['num_classes'], active_config['channels'], args.use_pretrained).to(DEVICE)
            print(f"Initialized {model_name} architecture.")

            # --- Establish Baseline Model (Once per dataset/model) ---
            baseline_model_loaded_or_trained = False
            if args.skip_train:
                if os.path.exists(model_save_path):
                    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
                    print(f"Loaded existing baseline model from '{model_save_path}' (--skip_train used).")
                    baseline_model_loaded_or_trained = True
                else:
                    print(f"Error: --skip_train specified but model file not found at '{model_save_path}'. Skipping model {model_name} for dataset {dataset_name}.")
                    continue # Skip this model for this dataset
            else: # args.skip_train is False
                if os.path.exists(model_save_path):
                    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
                    print(f"Loaded existing baseline model from '{model_save_path}' (found existing file, training skipped).")
                    baseline_model_loaded_or_trained = True
                else:
                    print(f"\n--- Training Baseline {model_name} on Clean {dataset_name.upper()} (No existing file found) ---")
                    optimizer = optim.SGD(model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)
                    scheduler = StepLR(optimizer, step_size=max(1, args.train_epochs // 3), gamma=0.1)
                    best_val_acc = 0.0
                    for epoch in range(args.train_epochs):
                        epoch_start_time = time.time()
                        # Use the base_train_loader and base_val_loader created outside the model loop
                        train_loss, train_acc = train_epoch(base_train_loader, model, criterion, optimizer, DEVICE)
                        val_loss, val_acc = test_epoch(base_val_loader, model, criterion, DEVICE, desc='Validation')
                        scheduler.step()
                        print(f"Epoch {epoch+1}/{args.train_epochs} | Time: {time.time()-epoch_start_time:.2f}s | "
                              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                        if val_acc > best_val_acc:
                            print(f"Validation accuracy improved ({best_val_acc:.2f}% -> {val_acc:.2f}%). Saving model...")
                            best_val_acc = val_acc
                            torch.save(model.state_dict(), model_save_path)
                    print("Baseline training finished.")

                    if os.path.exists(model_save_path):
                         model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
                         print(f"Loaded best baseline model from '{model_save_path}' after training.")
                         baseline_model_loaded_or_trained = True
                    else:
                         print(f"Error: Baseline model file not found even after training attempt at '{model_save_path}'. Skipping model {model_name} for dataset {dataset_name}.")
                         continue # Skip this model for this dataset

            if not baseline_model_loaded_or_trained:
                 print(f"Critical Error: Baseline model for {model_name}/{dataset_name} was neither loaded nor trained successfully. Skipping.")
                 continue # Skip this model for this dataset
            # --- End Baseline Handling ---

            # --- Evaluate Baseline Model (Once per dataset/model) ---
            print(f"\n--- Evaluating Established Baseline Model ({model_name}/{dataset_name}) ---")
            baseline_test_loss, baseline_test_acc = test_epoch(base_test_loader, model, criterion, DEVICE, desc='Clean Test (Full)')
            print(f"Baseline Model Clean Test Accuracy (Full): {baseline_test_acc:.2f}%")
            # Store baseline full result (RunIndex=-1 indicates it's a pre-run baseline)
            all_run_results.append({
                "RunIndex": -1, "Dataset": dataset_name, "Model": model_name, "FTRatio": -1.0,
                "DriftType": "Clean", "Metric": "BaselineFull", "Accuracy": baseline_test_acc,
                "RelevantCleanBaselineAcc": baseline_test_acc # Baseline for itself
            })

            baseline_target_subset_acc_val = 'N/A' # Default
            feature_drift_target_indices_set = active_config.get('feature_drift_target_indices', set())
            if feature_drift_target_indices_set:
                try:
                    if hasattr(clean_test_dataset, 'targets'):
                        clean_targets = np.array(clean_test_dataset.targets)
                    elif hasattr(clean_test_dataset, 'labels'):
                         clean_targets = np.array(clean_test_dataset.labels) % 10
                    else:
                        print("Warning: Cannot access targets/labels directly for clean test subset evaluation. Iterating (slow).")
                        clean_targets = np.array([clean_test_dataset[i][1] for i in range(len(clean_test_dataset))])
                        if dataset_name == 'svhn': clean_targets = clean_targets % 10

                    target_subset_indices_clean = [i for i, target in enumerate(clean_targets) if target in feature_drift_target_indices_set]

                    if target_subset_indices_clean:
                        clean_target_subset = Subset(clean_test_dataset, target_subset_indices_clean)
                        clean_target_subset_loader = DataLoader(clean_target_subset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS_REPRODUCIBLE)
                        print(f"Evaluating baseline model on the clean test subset corresponding to feature drift target classes ({len(target_subset_indices_clean)} samples)...")
                        baseline_target_subset_loss, baseline_target_subset_acc_val = test_epoch(clean_target_subset_loader, model, criterion, DEVICE, desc='Clean Test (Target Subset)')
                        print(f"Baseline Model Clean Test Accuracy (Target Subset Only): {baseline_target_subset_acc_val:.2f}%")
                    else:
                        print("Warning: No samples found in clean test set for feature drift target indices. Cannot calculate target subset baseline.")
                        baseline_target_subset_acc_val = 'N/A'
                except Exception as e:
                    print(f"Error during baseline evaluation on clean target subset: {e}")
                    baseline_target_subset_acc_val = 'Error'
            else:
                 baseline_target_subset_acc_val = 'N/A (No target indices defined)'

            # Store baseline subset result (RunIndex=-1 indicates it's a pre-run baseline)
            all_run_results.append({
                "RunIndex": -1, "Dataset": dataset_name, "Model": model_name, "FTRatio": -1.0,
                "DriftType": "Clean", "Metric": "BaselineFeatureTargetSubset", "Accuracy": baseline_target_subset_acc_val,
                "RelevantCleanBaselineAcc": baseline_target_subset_acc_val # Baseline for itself
            })
            # --- End Baseline Evaluations ---


            # --- Statistical Runs Loop ---
            for run_index in range(args.num_runs):
                current_seed = BASE_SEED + run_index + 1 # Start run index from 1 for seed offset
                print(f"\n--- Starting Run {run_index + 1}/{args.num_runs} with Seed {current_seed} ({model_name}/{dataset_name}) ---")
                random.seed(current_seed)
                np.random.seed(current_seed)
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(current_seed)
                # CUDNN flags remain True/False as set globally

                # --- Drift Type Loop ---
                drift_types = ['input', 'feature', 'output']
                for drift_type in drift_types:
                    print(f"\n  --- Processing Drift Type: {drift_type.upper()} (Run {run_index + 1}) ---")

                    # Determine relevant baseline for comparison *before* simulation
                    relevant_baseline_acc = 'N/A'
                    if drift_type == 'feature':
                        relevant_baseline_acc = baseline_target_subset_acc_val # Use the value calculated outside the loop
                    else: # input or output
                        relevant_baseline_acc = baseline_test_acc # Use the full baseline value calculated outside the loop

                    # Simulate drift *inside* the run loop for stochastic transforms
                    if drift_type == 'input':
                        source_drift_dataset, target_drift_dataset = simulate_input_drift(clean_test_dataset, active_config)
                        drift_title = "Input-Level Drift"
                    elif drift_type == 'feature':
                        source_drift_dataset, target_drift_dataset = simulate_feature_drift(clean_test_dataset, active_config)
                        drift_title = active_config['feature_drift_title']
                    elif drift_type == 'output':
                        source_drift_dataset, target_drift_dataset = simulate_output_drift(clean_test_dataset, active_config)
                        drift_title = active_config['output_drift_title']
                    else: continue

                    source_drift_loader = DataLoader(source_drift_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS_REPRODUCIBLE)
                    target_drift_loader = DataLoader(target_drift_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS_REPRODUCIBLE)

                    # Visualize only on first run if not skipped
                    if run_index == 0 and not args.skip_viz and len(source_drift_loader) > 0 and len(target_drift_loader) > 0:
                         viz_save_path = os.path.join(args.results_dir, f"{dataset_name}_{model_name}_{drift_type}_drift_viz.png") # Generic viz filename
                         visualize_drift(source_drift_loader, target_drift_loader, drift_title, active_config, viz_save_path)

                    # Evaluate the FIXED baseline model on this run's drifted data
                    print(f"  Evaluating baseline {model_name} on {drift_type} drifted data (Run {run_index + 1})...")
                    drift_loss, drift_acc = test_epoch(target_drift_loader, model, criterion, DEVICE, desc=f'{drift_type.capitalize()} Drift Test')
                    print(f"  Baseline Model {drift_type.capitalize()} Drift Test Accuracy: {drift_acc:.2f}%")
                    # Store baseline on drift result for this run
                    all_run_results.append({
                        "RunIndex": run_index + 1, "Dataset": dataset_name, "Model": model_name, "FTRatio": -1.0, # Indicate N/A ratio
                        "DriftType": drift_type, "Metric": "BaselineOnDrift", "Accuracy": drift_acc,
                        "RelevantCleanBaselineAcc": relevant_baseline_acc
                    })

                    # --- Fine-tuning Ratio Loop ---
                    for ft_ratio in args.ft_ratios:
                        print(f"\n    --- Processing FT Ratio: {ft_ratio:.2f} (Run {run_index + 1}, {drift_type}) ---")
                        num_target_samples = len(target_drift_dataset)
                        ft_subset_size = int(ft_ratio * num_target_samples)
                        if ft_subset_size == 0 and num_target_samples > 0: ft_subset_size = 1

                        if ft_subset_size > 0:
                            # Sampling uses the run-specific seed
                            ft_indices_local = random.sample(range(num_target_samples), ft_subset_size)
                            if isinstance(target_drift_dataset, Subset):
                                original_target_indices = target_drift_dataset.indices
                                ft_indices_global = [original_target_indices[i] for i in ft_indices_local]
                                ft_drift_subset = Subset(target_drift_dataset.dataset, ft_indices_global)
                            elif isinstance(target_drift_dataset, LabelShiftingDataset):
                                 base_for_ft = target_drift_dataset.base_dataset
                                 if isinstance(base_for_ft, Subset):
                                     original_target_indices = base_for_ft.indices
                                     ft_indices_global = [original_target_indices[i] for i in ft_indices_local]
                                     ft_drift_subset_base = Subset(base_for_ft.dataset, ft_indices_global)
                                     ft_drift_subset = LabelShiftingDataset(ft_drift_subset_base, target_drift_dataset.shift_map)
                                 else:
                                     ft_drift_subset_base = Subset(base_for_ft, ft_indices_local)
                                     ft_drift_subset = LabelShiftingDataset(ft_drift_subset_base, target_drift_dataset.shift_map)
                            else:
                                ft_drift_subset = Subset(target_drift_dataset, ft_indices_local)

                            # FT dataloader shuffle uses run-specific seed
                            ft_drift_loader = DataLoader(ft_drift_subset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS_REPRODUCIBLE, drop_last=True)
                            print(f"    Created fine-tuning subset with {len(ft_drift_subset)} samples for {drift_type} drift (Ratio: {ft_ratio}).")
                        else:
                            print(f"    Warning: Cannot create fine-tuning subset for {drift_type} drift (Ratio {ft_ratio}). Skipping fine-tuning for this ratio.")
                            # Optionally add placeholder results to CSV? Or just skip. Skipping for now.
                            continue

                        # --- Layer Group Fine-tuning Loop ---
                        layer_group_names = list(get_layer_groups(model, model_name).keys())
                        for group_name in layer_group_names:
                            print(f"\n      Fine-tuning group: {group_name} (Ratio: {ft_ratio}, Run {run_index + 1})")
                            # Always start FT from a fresh copy of the established baseline model
                            model_group_ft = copy.deepcopy(model)
                            for param in model_group_ft.parameters():
                                param.requires_grad = False
                            try:
                                current_group_generator = get_layer_groups(model_group_ft, model_name)[group_name]
                                params_to_tune = list(current_group_generator)
                            except Exception as e:
                                 print(f"        Error getting parameters for group '{group_name}': {e}. Skipping.")
                                 # Store error result for this run/ratio/group
                                 all_run_results.append({
                                     "RunIndex": run_index + 1, "Dataset": dataset_name, "Model": model_name, "FTRatio": ft_ratio,
                                     "DriftType": drift_type, "Metric": f"GroupFT_{group_name}", "Accuracy": "Error",
                                     "RelevantCleanBaselineAcc": relevant_baseline_acc
                                 })
                                 continue
                            if not params_to_tune:
                                print(f"        Warning: No parameters found for group {group_name}. Skipping.")
                                all_run_results.append({
                                     "RunIndex": run_index + 1, "Dataset": dataset_name, "Model": model_name, "FTRatio": ft_ratio,
                                     "DriftType": drift_type, "Metric": f"GroupFT_{group_name}", "Accuracy": 0.0, # Or 'N/A'
                                     "RelevantCleanBaselineAcc": relevant_baseline_acc
                                 })
                                continue

                            num_unfrozen = 0
                            for param in params_to_tune:
                                param.requires_grad = True
                                num_unfrozen += 1
                            trainable_params_in_model = list(filter(lambda p: p.requires_grad, model_group_ft.parameters()))
                            print(f"        Unfroze {len(trainable_params_in_model)} parameters for group {group_name}.")
                            if not trainable_params_in_model:
                                 print(f"        Warning: No trainable parameters found for group {group_name} after attempting unfreeze. Skipping.")
                                 all_run_results.append({
                                     "RunIndex": run_index + 1, "Dataset": dataset_name, "Model": model_name, "FTRatio": ft_ratio,
                                     "DriftType": drift_type, "Metric": f"GroupFT_{group_name}", "Accuracy": 0.0, # Or 'N/A'
                                     "RelevantCleanBaselineAcc": relevant_baseline_acc
                                 })
                                 continue

                            optimizer_ft_group = optim.SGD(trainable_params_in_model, lr=args.lr_ft, momentum=0.9, weight_decay=5e-4)
                            # Include run and ratio in FT model filename
                            ratio_percent_str_ft = str(int(ft_ratio*100))
                            best_ft_model_path_group = os.path.join(args.results_dir, f"{dataset_name}_{model_name}_{drift_type}_{group_name}_ft_run{run_index + 1}_ratio{ratio_percent_str_ft}.pth")
                            min_val_loss_ft = float('inf')
                            epochs_no_improve = 0

                            for epoch in range(args.ft_epochs):
                                model_group_ft.train()
                                # train_epoch uses ft_drift_loader which shuffles based on run-specific seed
                                ft_train_loss, ft_train_acc = train_epoch(ft_drift_loader, model_group_ft, criterion, optimizer_ft_group, DEVICE)
                                # test_epoch uses base_val_loader which is deterministic within a run
                                ft_val_loss, ft_val_acc = test_epoch(base_val_loader, model_group_ft, criterion, DEVICE, desc=f'FT Val ({group_name})')
                                # print(f"        FT Epoch {epoch+1}/{args.ft_epochs} | Train Loss: {ft_train_loss:.4f}, Acc: {ft_train_acc:.2f}% | Val Loss: {ft_val_loss:.4f}, Acc: {ft_val_acc:.2f}%") # Verbose
                                if ft_val_loss < min_val_loss_ft:
                                    min_val_loss_ft = ft_val_loss
                                    epochs_no_improve = 0
                                    torch.save(model_group_ft.state_dict(), best_ft_model_path_group)
                                else:
                                    epochs_no_improve += 1
                                    if epochs_no_improve >= args.early_stop_patience:
                                        # print(f"        Early stopping triggered after {epoch+1} epochs.") # Verbose
                                        break

                            # Evaluate the best fine-tuned model for this group/ratio/run
                            final_ft_acc = 'N/A'
                            if os.path.exists(best_ft_model_path_group):
                                model_group_ft.load_state_dict(torch.load(best_ft_model_path_group, map_location=DEVICE))
                                ft_group_test_loss, ft_group_test_acc = test_epoch(target_drift_loader, model_group_ft, criterion, DEVICE, desc=f'FT {group_name} Test ({drift_type})')
                                print(f"      Group '{group_name}' FT Model {drift_type.capitalize()} Drift Test Accuracy: {ft_group_test_acc:.2f}%")
                                final_ft_acc = ft_group_test_acc
                            else:
                                print(f"        Warning: Best group '{group_name}' fine-tuned model path not found. No FT result recorded.")
                                final_ft_acc = 'Error: NoModelSaved'

                            # Store fine-tuning result for this run/ratio/group
                            all_run_results.append({
                                "RunIndex": run_index + 1, "Dataset": dataset_name, "Model": model_name, "FTRatio": ft_ratio,
                                "DriftType": drift_type, "Metric": f"GroupFT_{group_name}", "Accuracy": final_ft_acc,
                                "RelevantCleanBaselineAcc": relevant_baseline_acc
                            })
                        # --- End Layer Group Loop ---
                    # --- End FT Ratio Loop ---
                # --- End Drift Type Loop ---
            # --- End Statistical Runs Loop ---
        # --- End Model Loop ---
    # --- End Dataset Loop ---

    # --- Final CSV Output ---
    print(f"\n{'='*20} Writing Combined Results Summary {'='*20}")
    # Use a generic filename for the combined results
    results_csv_path = os.path.join(args.results_dir, f"combined_drift_summary_runs{args.num_runs}.csv")
    csv_headers = ["RunIndex", "Dataset", "Model", "FTRatio", "DriftType", "Metric", "Accuracy", "RelevantCleanBaselineAcc"]

    try:
        with open(results_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
            # Format accuracy values before writing if they are floats
            formatted_results = []
            for row in all_run_results:
                new_row = row.copy()
                if isinstance(new_row["Accuracy"], (float, np.floating)):
                    new_row["Accuracy"] = f"{new_row['Accuracy']:.2f}"
                if isinstance(new_row["RelevantCleanBaselineAcc"], (float, np.floating)):
                    new_row["RelevantCleanBaselineAcc"] = f"{new_row['RelevantCleanBaselineAcc']:.2f}"
                formatted_results.append(new_row)
            writer.writerows(formatted_results)
        print(f"\nCombined results summary saved to: {results_csv_path}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")

    print("\n--- Experiment Complete ---")

# --- END OF FILE drift_exp.py ---