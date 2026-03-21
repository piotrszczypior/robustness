
## CLI Usage Reference

### 1. Setup Task
Prepares the environment and downloads necessary datasets.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | string | **Required** | Identifier of the dataset to prepare. |
| `--data-path` | string | `data/` | Root directory for storing datasets. |
| `--archives` | list | `None` | Specific archive filenames to download (e.g., `blur.tar`). |

---

### 2. Evaluate Task
Executes model evaluation across defined experiments and corruption types.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | string | `resnet152` | Model architecture to evaluate. |
| `--data-path` | string | `data/` | Path to the input dataset. |
| `--output-path` | string | `results/` | Directory for storing evaluation CSV results. |
| `--experiments` | string | `experiments/experiments.yaml` | Path to experiments configuration file. |
| `--run-batch`   | string | `None` | Filter to run a specific batch of experiments by name (e.g imagenet_c_blur) |
| `--run-single` | string | `None` | Filter to run a specific experiment by name. (e.g. imagenet_c_defocus_blur_3) |
| `--sync-drive` | flag | `False` | Enable automatic synchronization to Google Drive. |

---

### 3. Plot Task
Generates visualizations based on evaluation results and predefined recipes.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--plots` | string | `plots/plots.yaml` | Path to the plot specifications YAML. |
| `--recipes` | string | `plots/recipes.yaml` | Path to the plotting recipes YAML. |
| `--data` | string | `results/` | Directory containing source CSV data. |
| `--sync-drive` | flag | `False` | Enable automatic synchronization to Google Drive. |

---

## Constants & Defaults
Global defaults are managed in `src/const.py`:
- **Default Model**: `resnet152`
- **Default Data Path**: `data/`
- **Default Output Path**: `results/`
