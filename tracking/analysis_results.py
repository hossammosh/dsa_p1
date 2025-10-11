import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist
from lib.config.seqtrack.config import cfg, update_config_from_file

# --- Load YAML config ---
update_config_from_file("experiments/seqtrack/seqtrack_b256.yaml")

trackers = []
dataset_name = 'lasot'
trackers.extend(trackerlist(
    name='seqtrack',
    parameter_name='seqtrack_b256',
    dataset_name=dataset_name,
    run_ids=None,
    display_name='seqtrack_b256'
))

dataset = get_dataset(dataset_name)

# --- Filter sequences by YAML CLASSES if enabled ---
cls_cfg = cfg.DATA.TRAIN.CLASSES
if getattr(cls_cfg, "ENABLED", False):
    allowed = set(cls_cfg.NAMES)
    dataset = [seq for seq in dataset if seq.name.split('-')[0] in allowed]
    print(f"✅ Evaluating only classes: {allowed} ({len(dataset)} sequences)")
else:
    print("✅ Class filtering disabled — evaluating all sequences.")

# --- Run evaluation ---
print_results(
    trackers,
    dataset,
    dataset_name,
    merge_results=True,
    plot_types=('success', 'prec', 'norm_prec'),
    force_evaluation=True
)
