import os
import glob
import torch
import random
import numpy as np
import traceback
from lib.train.admin import multigpu
from torch.utils.data.distributed import DistributedSampler
from lib.train.run_training import init_seeds
import lib.train.data_recorder as data_recorder


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def _write_to_log(self, message):
        """Helper method to write messages to the log file if it exists."""
        if hasattr(self.settings, 'log_file') and self.settings.log_file:
            try:
                with open(self.settings.log_file, 'a') as f:
                    f.write(message + '\n')
            except Exception as e:
                print(f"Error writing to log file {self.settings.log_file}: {e}", flush=True)

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.settings.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)
        self.settings = settings

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            '''2021.1.4 New function: specify checkpoint dir'''
            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            else:
                self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
            print("checkpoints will be saved to %s" % self._checkpoint_dir)

            if self.settings.local_rank in [-1, 0]:
                if not os.path.exists(self._checkpoint_dir):
                    print("Training with multiple GPUs. checkpoints directory doesn't exist. "
                          "Create checkpoints directory")
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def train(self, max_epochs, load_latest=False, fail_safe=True, load_previous_ckpt=False, distill=False):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        num_tries = 1
        for i in range(num_tries):
            try:
                # if load_latest:
                # self.load_checkpoint(self.settings.selected_sampling_epoch)
                if load_latest:
                    latest = 0
                    start_epoch = 1
                    # Construct the full checkpoint path
                    checkpoint_base = getattr(self, '_checkpoint_dir', '')
                    checkpoint_full_path = os.path.join(checkpoint_base, 'train', 'seqtrack', 'seqtrack_b256')

                    if os.path.exists(checkpoint_full_path):
                        import re

                        # Find all checkpoint files and their epochs
                        checkpoints = [
                            {
                                'filename': f,
                                'epoch': int(match.group(1)),
                                'path': os.path.join(checkpoint_full_path, f)
                            }
                            for f in os.listdir(checkpoint_full_path)
                            if (f.endswith(('.pth.tar', '.pth')) and
                                (match := re.search(r'_ep(\d+)\.pth(\.tar)?$', f)))
                        ]

                        if checkpoints:
                            # Find the latest checkpoint by epoch
                            latest = max(checkpoints, key=lambda x: x['epoch'])
                            start_epoch = max(1, latest['epoch'] + 1)
                            print(f"Found latest checkpoint: {latest['filename']} (epoch {latest['epoch']})")
                            self.load_checkpoint(latest['path'])
                        else:
                            print(f"No valid checkpoints found in {checkpoint_full_path}. Starting from scratch.")
                    else:
                        print(f"Checkpoint directory {checkpoint_full_path} not found. Starting from scratch.")

                if load_previous_ckpt:
                    # Define the checkpoint directory
                    checkpoint_dir = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
                    if isinstance(load_previous_ckpt, int):
                        # Load specific epoch checkpoint
                        epoch_num = load_previous_ckpt
                        checkpoint_path = os.path.join(checkpoint_dir, f'SEQTRACK_ep{epoch_num:04d}.pth.tar')

                        if os.path.exists(checkpoint_path):
                            print(f"Loading checkpoint for epoch {epoch_num}: {checkpoint_path}")
                            self.load_checkpoint(load_previous_ckpt)
                            start_epoch = load_previous_ckpt + 1
                            print(f"Resuming training from epoch {start_epoch}")
                        else:
                            print(f"Checkpoint for epoch {epoch_num} not found at {checkpoint_path}")
                    else:
                        # Original behavior for boolean True
                        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'SEQTRACK_ep*.pth.tar')))

                        if checkpoint_files:
                            latest_checkpoint = checkpoint_files[-1]
                            print(f"Loading latest checkpoint: {latest_checkpoint}")
                            self.load_checkpoint(latest_checkpoint)
                            start_epoch = int(latest_checkpoint.split('_ep')[-1].split('.pth.tar')[0]) + 1
                            print(f"Resuming training from epoch {start_epoch}")
                        else:
                            print(f"No checkpoints found in {checkpoint_dir}. Starting from epoch {start_epoch}")

                # Load teacher model if distillation is enabled
                if distill:
                    directory_teacher = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_teacher)
                    self.load_state_dict(directory_teacher, distill=True)

                for epoch in range(start_epoch, max_epochs + 1):
                    self.settings.epoch = epoch
                    data_recorder.set_epoch(settings=self.settings)

                    if (self.settings.selected_sampling and epoch >= self.settings.selected_sampling_epoch):
                        self.settings.top_selected_samples = int(
                            self.settings.top_sample_ratio * len(self.loaders[0].dataset))
                        self.loaders[0].dataset.load_selected_samples()
                        data_recorder.set_sampling(self.settings.selected_sampling)

                    init_seeds(42)
                    print('epoch no.= ', epoch, " at base trainer epoch loop")
                    self.train_epoch()
                    if self.lr_scheduler is not None:
                        if self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)
                    checkpoint_save_interval = self.settings.checkpoint_save_interval
                    if epoch % checkpoint_save_interval == 0 or (
                            epoch == max_epochs and max_epochs % checkpoint_save_interval != 0):
                        if self._checkpoint_dir:
                            if self.settings.local_rank in [-1, 0]:
                                self.save_checkpoint()
            except:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.settings.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise
        print('base trainer  Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    def _get_rng_states(self):
        """Get current RNG states for all random number generators."""
        states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark
        }
        return states

    def _set_rng_states(self, states):
        """Set RNG states from a saved state dictionary."""
        if states is None:
            return

        if 'python' in states:
            random.setstate(states['python'])
        if 'numpy' in states:
            np.random.set_state(states['numpy'])
        if 'torch' in states:
            torch.set_rng_state(states['torch'])
        if torch.cuda.is_available() and 'torch_cuda' in states and states['torch_cuda'] is not None:
            torch.cuda.set_rng_state_all(states['torch_cuda'])
        if 'cudnn_deterministic' in states:
            torch.backends.cudnn.deterministic = states['cudnn_deterministic']
        if 'cudnn_benchmark' in states:
            torch.backends.cudnn.benchmark = states['cudnn_benchmark']

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables, including optimizer, scheduler, and RNG states."""
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net
        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        # Prepare state dict with all necessary components for perfect resume
        state = {
            'net': net.state_dict(),
            'net_info': getattr(net, 'config', None),
            'constructor': getattr(net, 'constructor', None),
            'net_settings': getattr(net, 'settings', None),
            'actor_type': actor_type,
            'net_type': net_type,
            'stats': self.stats,
            'epoch': self.settings.epoch,
            'has_checkpoint': True,
            # Optimizer state
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            'optimizer_class': self.optimizer.__class__.__name__ if self.optimizer else None,
            # Learning rate scheduler state
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'lr_scheduler_class': self.lr_scheduler.__class__.__name__ if self.lr_scheduler else None,
            # RNG states for reproducibility
            'rng_states': self._get_rng_states(),
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save to temporary file first, then rename for atomic operation
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.settings.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.settings.epoch)
        os.rename(tmp_file_path, file_path)

        print(f"Checkpoint saved: {file_path}")

    def load_checkpoint(self, checkpoint=None, fields=None, ignore_fields=None, load_constructor=False):
        """Loads checkpoint with optimizer, scheduler, and RNG states for perfect resume."""
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net
        net_type = type(net).__name__

        # Resolve checkpoint path
        if checkpoint is None:
            checkpoint_dir = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
            checkpoint_list = sorted(glob.glob(f'{checkpoint_dir}/{net_type}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            checkpoint_dir = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
            checkpoint_path = os.path.join(checkpoint_dir, f'SEQTRACK_ep{checkpoint:04d}.pth.tar')
        elif isinstance(checkpoint, str):
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint file not found: {checkpoint_path}')
            return

        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        print(f"Loading checkpoint from: {checkpoint_path}")

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

        # Load network state
        if 'net' in checkpoint_dict and 'net' not in ignore_fields:
            net.load_state_dict(checkpoint_dict['net'])
            print("Network state loaded")

        # Load optimizer state
        if ('optimizer' in checkpoint_dict and checkpoint_dict['optimizer'] is not None and
            self.optimizer is not None and 'optimizer' not in ignore_fields):
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
            # Move optimizer state tensors to correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print("Optimizer state loaded")

        # Load learning rate scheduler state
        if ('lr_scheduler' in checkpoint_dict and checkpoint_dict['lr_scheduler'] is not None and
            self.lr_scheduler is not None and 'lr_scheduler' not in ignore_fields):
            self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
            print("Learning rate scheduler state loaded")

        # Load RNG states for reproducibility
        if 'rng_states' in checkpoint_dict and checkpoint_dict['rng_states'] is not None and 'rng_states' not in ignore_fields:
            self._set_rng_states(checkpoint_dict['rng_states'])
            print("RNG states restored")

        # Load other fields
        for key in fields:
            if key in ignore_fields or key in ['net', 'optimizer', 'lr_scheduler', 'rng_states']:
                continue
            if key in checkpoint_dict:
                setattr(self, key, checkpoint_dict[key])

        # Restore epoch
        if 'epoch' in checkpoint_dict:
            self.settings.epoch = checkpoint_dict['epoch']
            print(f"Resuming from epoch {self.settings.epoch}")

        # Restore constructor if needed
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        print(f"Checkpoint loaded successfully. Training will resume from epoch {self.settings.epoch + 1}")

    def load_state_dict(self, checkpoint=None, distill=False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        if distill:
            net = self.actor.net_teacher.module if multigpu.is_multi_gpu(self.actor.net_teacher) \
                else self.actor.net_teacher
        else:
            net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__

        if isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        print("Loading pretrained model from ", checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        missing_k, unexpected_k = net.load_state_dict(checkpoint_dict["net"], strict=False)
        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)

        return True
