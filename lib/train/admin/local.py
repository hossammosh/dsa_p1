import os
class EnvironmentSettings:
    def __init__(self):
        # self.workspace_dir = '/mnt/e/current_research/seq'    # Base directory for saving network checkpoints.
        # self.tensorboard_dir = '/mnt/e/current_research/seq/tensorboard'    # Directory for tensorboard files.
        # self.pretrained_networks = '/mnt/e/current_research/seq/pretrained_networks'
        self.workspace_dir = os.getcwd()  # This will be '/mnt/e/current_research/reducedDS'
        self.tensorboard_dir = os.path.join(self.workspace_dir, 'tensorboard')
        self.pretrained_networks = os.path.join(self.workspace_dir, 'pretrained_networks')
        self.checkpoints_dir = os.path.join(self.workspace_dir, 'pretrained_networks')
        #checkpoints
        self.lasot_dir = os.path.join(os.getcwd(), 'data', 'lasot')

