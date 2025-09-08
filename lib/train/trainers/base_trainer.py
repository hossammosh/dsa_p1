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
                    latest=0
                    start_epoch=1
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
                    checkpoint_save_interval  = self.settings.checkpoint_save_interval
                    if epoch % checkpoint_save_interval == 0 or (epoch == max_epochs and max_epochs % checkpoint_save_interval != 0):
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
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'cudnn': torch.backends.cudnn.deterministic
        }
        
    def _set_rng_states(self, states):
        """Set RNG states from a saved state dictionary."""
        if states is None:
            return
            
        random.setstate(states['python'])
        np.random.set_state(states['numpy'])
        torch.set_rng_state(states['torch'])
        if torch.cuda.is_available() and states['cuda'] is not None:
            torch.cuda.set_rng_state(states['cuda'])
        torch.backends.cudnn.deterministic = states['cudnn']
        
    def _log_rng_states(self, prefix=""):
        """Log current RNG states to file."""
        states = self._get_rng_states()
        log_entries = [f"\n=== {prefix}RNG States ==="]
        
        try:
            # Log Python RNG state
            log_entries.append("Python RNG state:")
            log_entries.append(f"  - First number: {random.random()}")
            
            # Log NumPy RNG state
            log_entries.append("\nNumPy RNG state:")
            log_entries.append(f"  - First number: {np.random.random()}")
            
            # Log PyTorch RNG state
            log_entries.append("\nPyTorch RNG state:")
            log_entries.append(f"  - First number: {torch.rand(1).item()}")
            
            # Log CUDA RNG state if available
            if torch.cuda.is_available():
                log_entries.append("\nCUDA RNG state:")
                # Use torch.cuda.FloatTensor instead of torch.tensor to avoid deprecation warning
                log_entries.append(f"  - First number: {torch.cuda.FloatTensor(1).normal_().item()}")
                
            log_entries.append(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
            
        except Exception as e:
            log_entries.append(f"Error logging RNG states: {str(e)}")
            
        log_str = "\n".join(log_entries)
        print(log_str, flush=True)
        self._write_to_log(log_str)
        
        # Return to previous RNG states
        self._set_rng_states(states)
        
    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""
        # Log RNG states before saving
        self._log_rng_states("Before saving checkpoint: ")
        
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net
        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        # Log training state before saving
        log_entries = []
        
        # Log LR scheduler state
        if self.lr_scheduler is not None:
            log_entries.extend([
                "\n=== LR Scheduler State ===",
                f"Last Epoch: {getattr(self.lr_scheduler, 'last_epoch', 'N/A')}",
                f"State Dict: {self.lr_scheduler.state_dict()}"
            ])
            
        # Add RNG states to log
        rng_states = self._get_rng_states()
        log_entries.append("\n=== RNG States ===")
        
        # Log Python RNG state
        if rng_states['python'] and len(rng_states['python']) > 1 and len(rng_states['python'][1]) > 0:
            log_entries.append(f"Python RNG state: {rng_states['python'][1][0]}")
        else:
            log_entries.append("Python RNG state: N/A")
            
        # Log NumPy RNG state
        if rng_states['numpy'] and len(rng_states['numpy']) > 1 and len(rng_states['numpy'][1]) > 0:
            log_entries.append(f"NumPy RNG state: {rng_states['numpy'][1][0]}")
        else:
            log_entries.append("NumPy RNG state: N/A")
            
        # Log PyTorch RNG state (first few bytes as hex)
        if rng_states['torch'] is not None:
            try:
                # Convert tensor to bytes and get first 8 bytes as hex
                torch_bytes = rng_states['torch'].numpy().tobytes()[:8]
                log_entries.append(f"PyTorch RNG state: {torch_bytes.hex()}...")
            except Exception as e:
                log_entries.append(f"PyTorch RNG state: Error ({str(e)})")
        else:
            log_entries.append("PyTorch RNG state: N/A")
            
        # Log CUDA RNG state if available
        if torch.cuda.is_available():
            if rng_states['cuda'] is not None:
                try:
                    # Convert CUDA tensor to bytes and get first 8 bytes as hex
                    cuda_bytes = rng_states['cuda'].cpu().numpy().tobytes()[:8]
                    log_entries.append(f"CUDA RNG state: {cuda_bytes.hex()}...")
                except Exception as e:
                    log_entries.append(f"CUDA RNG state: Error ({str(e)})")
            else:
                log_entries.append("CUDA RNG state: N/A")
        else:
            log_entries.append("CUDA not available")
            
        log_entries.append(f"cuDNN deterministic: {rng_states['cudnn']}")

        # Get optimizer state for logging
        optimizer_state = None
        if self.optimizer is not None:
            optimizer_state = self.optimizer.state_dict()
            log_entries.append("\n=== Optimizer State ===")
            log_entries.append(f"Optimizer class: {self.optimizer.__class__.__name__}")
            log_entries.append(f"Number of parameter groups: {len(optimizer_state['param_groups'])}")
            
            # Log learning rate for each parameter group
            for i, group in enumerate(optimizer_state['param_groups']):
                log_entries.append(f"  Group {i} - lr: {group.get('lr', 'N/A')}, weight_decay: {group.get('weight_decay', 'N/A')}")
            
            # Log detailed state information
            if optimizer_state['state']:
                log_entries.append("\nOptimizer Internal State (first 3 parameters):")
                state_keys = set()
                total_params = len(optimizer_state['state'])
                
                for param_idx, (param_id, state) in enumerate(optimizer_state['state'].items()):
                    if param_idx < 3:  # Only show first 3 parameters to avoid log spam
                        log_entries.append(f"\n  Parameter {param_idx} state:")
                        for key, value in state.items():
                            state_keys.add(key)
                            if torch.is_tensor(value):
                                log_entries.append(f"    {key}: shape={value.shape}, dtype={value.dtype}, mean={value.mean().item():.6f}, std={value.std().item():.6f}")
                            else:
                                log_entries.append(f"    {key}: {value}")
                    else:
                        # Still collect state keys from other parameters
                        state_keys.update(state.keys())
                
                log_entries.append(f"\nOptimizer state keys: {', '.join(state_keys) if state_keys else 'None'}")
                log_entries.append(f"Total parameters with state: {total_params}")
            else:
                log_entries.append("No optimizer state available (not yet initialized?)")
        
        if hasattr(self.lr_scheduler, 'get_last_lr'):
            log_entries.append(f"\nCurrent Learning Rate: {self.lr_scheduler.get_last_lr()}")
        log_entries.append("\n" + "="*33)
        
        # Print to console and write to log file
        log_str = "\n".join(log_entries)
        print(log_str, flush=True)
        self._write_to_log(log_str)

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
            'rng_states': self._get_rng_states(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
            'optimizer': optimizer_state,
            'optimizer_class': self.optimizer.__class__.__name__ if self.optimizer else None
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        print(directory)
        if not os.path.exists(directory):
            print("directory doesn't exist. creating...")
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.settings.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.settings.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)

    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
                
        Also restores RNG states if they exist in the checkpoint.
        """
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net
        net_type = type(net).__name__
        #checkpoint_dir = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        #checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,self.settings.project_path, net_type)))

        if checkpoint is None:
            checkpoint_dir = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
            checkpoint_list = sorted(
                glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir, self.settings.project_path, net_type)))

            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_dir = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
            checkpoint_path = os.path.join(checkpoint_dir, f'SEQTRACK_ep{checkpoint:04d}.pth.tar')
        elif isinstance(checkpoint, str):
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
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']
        #
        #     # Never load the scheduler. It exists in older checkpoints.
        # ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        #fields = list(fields)
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer' and checkpoint_dict[key] is not None and self.optimizer is not None:
                log_entries = ["\n=== Loading Optimizer State ==="]
                saved_optimizer = checkpoint_dict[key]
                
                # Log saved optimizer info
                log_entries.append(f"Saved optimizer class: {checkpoint_dict.get('optimizer_class', 'Unknown')}")
                log_entries.append(f"Current optimizer class: {self.optimizer.__class__.__name__}")
                
                # Log parameter groups info
                if 'param_groups' in saved_optimizer:
                    log_entries.append(f"Number of parameter groups in checkpoint: {len(saved_optimizer['param_groups'])}")
                    for i, group in enumerate(saved_optimizer['param_groups']):
                        log_entries.append(f"  Group {i} - lr: {group.get('lr', 'N/A')}, weight_decay: {group.get('weight_decay', 'N/A')}")
                
                # Log state info
                if 'state' in saved_optimizer:
                    state_keys = set()
                    for param_id, state in saved_optimizer['state'].items():
                        state_keys.update(state.keys())
                    log_entries.append(f"Saved optimizer state keys: {', '.join(state_keys) if state_keys else 'None'}")
                
                try:
                    # Load the optimizer state
                    self.optimizer.load_state_dict(saved_optimizer)
                    log_entries.append("Successfully loaded optimizer state")
                    
                    # Move optimizer state to the correct device
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                    
                    # Log current optimizer state after loading
                    current_state = self.optimizer.state_dict()
                    log_entries.append(f"Current optimizer learning rates: {[g['lr'] for g in current_state['param_groups']]}")
                    
                except Exception as e:
                    log_entries.append(f"Error loading optimizer state: {str(e)}")
                
                # Print to console and write to log file
                log_str = "\n".join(log_entries)
                print(log_str, flush=True)
                self._write_to_log(log_str)
            elif key == 'lr_scheduler' and checkpoint_dict[key] is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler and settings
        if 'epoch' in fields:
            log_entries = ["\n=== Before Loading Checkpoint ==="]
            if self.lr_scheduler is not None:
                log_entries.extend([
                    "Current LR Scheduler State:",
                    f"Last Epoch: {getattr(self.lr_scheduler, 'last_epoch', 'N/A')}",
                    f"State Dict: {self.lr_scheduler.state_dict() if hasattr(self.lr_scheduler, 'state_dict') else 'N/A'}"
                ])
                if hasattr(self.lr_scheduler, 'get_last_lr'):
                    log_entries.append(f"Current Learning Rate: {self.lr_scheduler.get_last_lr()}")
            
            # Print to console and write to log file
            log_str = "\n".join(log_entries)
            print(log_str, flush=True)
            self._write_to_log(log_str)
            
            self.settings.epoch = checkpoint_dict['epoch']  # Restore the epoch from checkpoint
            
            if self.lr_scheduler is not None:
                # First load the scheduler state if available
                if 'lr_scheduler' in checkpoint_dict and checkpoint_dict['lr_scheduler'] is not None:
                    saved_state = checkpoint_dict['lr_scheduler']
                    log_entries = [
                        "\n=== Loading LR Scheduler State ===",
                        f"Saved State: {saved_state}"
                    ]
                    
                    # Check if _step_count exists in the saved state
                    has_step_count = '_step_count' in saved_state
                    log_entries.append(f"Checkpoint contains _step_count: {has_step_count}")
                    
                    if has_step_count:
                        log_entries.append(f"Saved _step_count value: {saved_state['_step_count']}")
                    
                    # Load the state
                    self.lr_scheduler.load_state_dict(saved_state)
                    
                    # Log the state after loading
                    current_state = self.lr_scheduler.state_dict()
                    log_entries.append(f"State after load: {current_state}")
                    log_entries.append(f"Current _step_count in scheduler: {current_state.get('_step_count', 'Not found')}")
                    
                    # Print to console and write to log file
                    log_str = "\n".join(log_entries)
                    print(log_str, flush=True)
                    self._write_to_log(log_str)
                
                # Ensure the last_epoch is properly set
                self.lr_scheduler.last_epoch = self.settings.epoch
                
                log_entries = [
                    "\n=== After Loading Checkpoint ===",
                    f"Last Epoch set to: {self.lr_scheduler.last_epoch}",
                    f"Final State: {self.lr_scheduler.state_dict()}"
                ]
                if hasattr(self.lr_scheduler, 'get_last_lr'):
                    log_entries.append(f"Current Learning Rate: {self.lr_scheduler.get_last_lr()}")
                log_entries.append("=================================\n")
                
                # Print to console and write to log file
                log_str = "\n".join(log_entries)
                print(log_str, flush=True)
                self._write_to_log(log_str)
            
            # Update the epoch in data_samplers
            for loader in self.loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.settings.epoch)
        
        # Restore RNG states if they exist in the checkpoint
        if 'rng_states' in checkpoint_dict:
            self._set_rng_states(checkpoint_dict['rng_states'])
            print("Restored RNG states from checkpoint")
            # Log the restored RNG states
            self._log_rng_states("After restoring RNG states: ")
        
        return True

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
