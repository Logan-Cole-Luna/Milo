# README.md

# UAV Detection Experimentation (`experiments/temporal`)

This directory contains the core components for training and evaluating UAV (Unmanned Aerial Vehicle) detection models, with a special focus on leveraging temporal information from video sequences. The primary script, `train_uav_detector.py`, orchestrates the entire process, from data loading and preprocessing to model training, evaluation, and results visualization.

## Core Components and Workflow

1.  **Configuration (`config_uav.py`):**
    *   This central file (located in the project root) dictates all experimental parameters. Key settings include:
        *   `MODEL_TYPE`: Specifies whether to use a "single_frame" or "temporal" model.
        *   `SEQ_LEN`: Defines the number of consecutive frames for temporal models.
        *   `DATA_ROOT`: Path to the UAV dataset.
        *   `OPTIMIZERS`, `LEARNING_RATE`, `EPOCHS`, `BATCH_SIZE`: Standard training hyperparameters.
        *   `RESULTS_BASE_DIR`: Where to save experiment outputs (logs, plots, model weights if implemented).
        *   `VISUALIZATION_SAMPLES`: Number of samples to visualize with predictions.
        *   `OVERFIT`: Flag to run in overfitting mode on a small subset of data.

2.  **Main Training Script (`train_uav_detector.py`):
    *   **Orchestration:** This is the entry point for running experiments. It initializes settings, pre-loads datasets, and then utilizes `experiment_runner.py` to manage multiple runs (e.g., for different optimizers or configurations).
    *   **Dataset Pre-loading:** To optimize repeated runs, datasets (train, validation, test) are loaded once at the beginning of the script's execution.
    *   **Experiment Iteration (`train_uav_experiment_iteration`):** This function is the core workhorse called by `experiment_runner.py` for each individual experiment run. It handles:
        *   Model instantiation (either `ResNet50Regressor` for single frames or `TemporalResNet50Regressor` for sequences, based on `C.MODEL_TYPE`).
        *   Setting up DataLoaders using the pre-loaded datasets.
        *   Defining loss functions (Smooth L1 + IoU-based loss), optimizer, and learning rate scheduler.
        *   Executing the training and validation loops via `run_epoch`.
        *   Performing a final evaluation on the test set.
        *   Collecting and returning detailed metrics (per-iteration costs, wall times, epoch-wise train/val losses and IoUs, test metrics) for `experiment_runner.py`.
        *   Managing visualizations of predictions via `visualize_samples`.
    *   **Epoch Execution (`run_epoch`):** Conducts a single pass through the data (an epoch) for either training or evaluation. It computes losses, updates model weights (if training), and aggregates metrics.
    *   **Visualization (`visualize_samples`):** Generates images with predicted and ground-truth bounding boxes overlaid, aiding in qualitative assessment of model performance.
    *   **Reproducibility:** Sets random seeds for `random`, `numpy`, and `torch` to ensure consistent results.
    *   **Results Management:** Creates structured directories for saving experiment outputs, including configuration logs, metric CSVs, and plots.

3.  **Dataset Handling (`uav_dataset.py` - `FrameBBOXDataset`):
    *   Provides a custom PyTorch `Dataset` class for the UAV data.
    *   Handles loading of images and corresponding YOLO-formatted bounding box labels.
    *   Supports both single-frame and temporal sequence loading (controlled by `is_temporal` and `seq_len` arguments, derived from `config_uav.py`).
    *   Integrates with Albumentations for image transformations (resize, normalize, convert to tensor).
    *   Includes optimizations like pre-filtering valid label files to speed up initialization.

4.  **Model Architectures (`network.py`):
    *   `ResNet50Regressor`: A ResNet-50 based model adapted for bounding box regression (outputting 4 values: cx, cy, w, h) from single image frames.
    *   `TemporalResNet50Regressor`: An extension or modification of the ResNet-50 architecture designed to process sequences of frames (`C.SEQ_LEN` frames) to predict the bounding box for the target (last) frame. The internal mechanism for aggregating temporal information (e.g., 3D convolutions, recurrent layers, attention) is a key aspect of this model.

5.  **Metrics (`uav_metrics.py`):
    *   `bbox_iou`: Calculates the Intersection over Union (IoU) between predicted and ground-truth bounding boxes. This is a primary metric for evaluating object detection performance.

6.  **Experiment Management (`experiment_runner.py`, `plotting.py` - from parent `experiments` module):
    *   `experiment_runner.py`: A generic utility that takes a training function (like `train_uav_experiment_iteration`), a list of optimizers/configurations, and manages multiple runs, collecting results and generating comparative plots and CSV summaries.
    *   `plotting.py`: Provides functions for generating various plots (e.g., loss/accuracy vs. epoch, iteration cost vs. walltime) in a consistent style.

## Running an Experiment

1.  **Configure:** Modify `config_uav.py` in the project root to set desired parameters (dataset paths, model type, hyperparameters, etc.).
2.  **Execute:** Run the main training script from the project root directory:
    ```bash
    python experiments/temporal/train_uav_detector.py
    ```
3.  **Results:** Outputs (logs, CSV files with metrics, plots, visualizations) will be saved in the directory specified by `C.RESULTS_BASE_DIR` in `config_uav.py`, under a subfolder named after `C.EXPERIMENT_NAME`.

## Key Features & Recent Improvements

*   **Temporal vs. Single-Frame Mode:** Easily switchable via `C.MODEL_TYPE`.
*   **Global Dataset Loading:** Datasets are loaded once, speeding up multiple experiment runs within a single script execution.
*   **Modular Design:** Code is organized into specific files for dataset handling, model definitions, metrics, and training orchestration.
*   **Integration with `experiment_runner`:** Allows for systematic comparison of different optimizers or configurations.
*   **Detailed Metric Collection:** Captures per-iteration costs and batch processing times for fine-grained performance analysis (e.g., walltime plots).
*   **Optimized Dataset Initialization:** `FrameBBOXDataset` pre-filters label files to avoid unnecessary image processing for frames without valid annotations.
*   **Comprehensive Logging:** Experiment settings and detailed metrics are saved for traceability and later analysis.
*   **Visualization:** Provides visual feedback on model predictions.

This structured approach facilitates robust experimentation and clear evaluation of different strategies for UAV detection.

## Temporal Processing for UAV Detection: A Deeper Dive

The integration of temporal processing represents a significant advancement in our UAV detection methodology. This section delves into the underlying theory, our specific implementation within this project, how it seamlessly fits into the existing architecture, and the expected performance enhancements.

### I. The Theory: Why Temporal Information Matters for Object Detection

Traditional object detection models process images in isolation. While effective for many static scenes, this approach discards valuable information present in video sequences, especially for dynamic objects like UAVs. Temporal processing leverages the relationship between consecutive frames to build a richer understanding of the scene.

1.  **Motion Cues as a Feature:**
    *   **Velocity and Trajectory:** The movement pattern of an object is a powerful discriminant. UAVs often exhibit characteristic flight paths and speeds. Temporal models can learn these motion signatures (e.g., smooth flight vs. erratic movements of birds) to distinguish targets from distractors.
    *   **Disambiguation:** An object that is ambiguous in a single frame (e.g., a distant speck) can be more confidently identified by observing its motion consistency over several frames.

2.  **Enhanced Contextual Understanding:**
    *   **Occlusion Handling:** If a UAV is partially or briefly occluded in one frame, its presence and approximate location can be inferred from preceding and succeeding frames where it was visible.
    *   **Appearance Variation:** UAVs can appear differently depending on viewing angle, lighting, and distance. Observing these changes over a short time window helps the model build a more robust appearance model.

3.  **Improved Temporal Consistency and Reduced Flicker:**
    *   Frame-by-frame independent detections can lead to "flickering" boxes—where a detection appears, disappears, and reappears—even if the object is continuously present. Temporal models, by considering past information, tend to produce smoother, more stable detection sequences.

### II. How It Works: Our Implementation Strategy

We've implemented temporal processing by adapting our dataset handling, model architecture, and training pipeline to process sequences of frames.

1.  **Configuration (`config_uav.py`):
    *   **Activation:** The temporal pipeline is activated by setting `C.MODEL_TYPE = "temporal"` in the configuration file.
    *   **Sequence Length (`C.SEQ_LEN`):** This crucial parameter defines how many consecutive frames are processed as a single input sample by the model. For instance, if `C.SEQ_LEN = 5`, the model receives a "cuboid" of 5 frames.

2.  **Dataset Adaptation (`uav_dataset.py` - `FrameBBOXDataset`):
    *   **Sequence Grouping:** When `is_temporal=True` (set based on `C.MODEL_TYPE`), the `FrameBBOXDataset` is responsible for gathering `C.SEQ_LEN` consecutive image frames.
    *   **Label Association:** The bounding box annotation for the *last frame* in the sequence is associated with the entire sequence. This is a common strategy, where the model uses the context of the preceding frames to make a prediction for the current (last) frame.
    *   **Data Integrity:** The dataset loader includes logic to ensure that frames within a sequence are indeed consecutive and belong to the same video segment (e.g., by checking filename patterns).
    *   **Output Tensor:** The data loader yields batches where the image tensor has a shape of `(BatchSize, SEQ_LEN, Channels, Height, Width)`, representing a batch of frame sequences.

3.  **Temporal Model Architecture (`network.py` - `TemporalResNet50Regressor`):
    *   **Input Processing:** The `TemporalResNet50Regressor` is specifically designed to accept these 5D tensors `(B, S, C, H, W)`.
    *   **Feature Extraction:** Internally, this model employs mechanisms (which could range from 3D convolutions that operate directly on the spatio-temporal volume, to recurrent layers like LSTMs/GRUs processing frame-wise features, or attention mechanisms that weigh the importance of different frames/features over time) to extract relevant spatio-temporal features.
    *   **Prediction:** The model then uses these aggregated features to regress the bounding box coordinates (`cx, cy, w, h`) for the target frame (the last frame of the input sequence).

4.  **Training and Orchestration (`train_uav_detector.py`):
    *   **Conditional Instantiation:** The main script checks `C.MODEL_TYPE` at runtime. If set to "temporal", it instantiates the `TemporalResNet50Regressor` and configures `FrameBBOXDataset` to operate in temporal mode with the specified `C.SEQ_LEN`.
    *   **Training Loop (`run_epoch`):** The core training loop remains largely the same, but the model now processes sequences. Loss (e.g., L1 + IoU loss) is calculated based on the prediction for the target frame against its ground truth.
    *   **Visualization (`visualize_samples`):** For temporal models, the visualization function loads the full sequence required by the model for inference. The predicted bounding box is then drawn on the target (last) frame of that sequence, alongside its ground truth label.

### III. Integration with Existing Project Structure

A key design goal was to integrate temporal capabilities with minimal disruption to the established experimental framework.

*   **`experiment_runner.py` Compatibility:** The `train_uav_experiment_iteration` wrapper function abstracts the differences between single-frame and temporal runs. The `experiment_runner.py` can thus manage, compare, and plot results from both types of models using the same interface, as the returned metrics (loss, IoU, walltime, etc.) are consistent.
*   **Configuration-Driven:** The switch between temporal and single-frame processing is controlled by a single configuration flag (`C.MODEL_TYPE`), making it easy to switch between modes for comparative experiments.
*   **Modular Helper Files:** The temporal logic is encapsulated within `uav_dataset.py` (for data loading) and `network.py` (for the model architecture), adhering to the project's modular design.

### IV. Anticipated Performance Improvements

By incorporating temporal information, we expect several key performance benefits for UAV detection:

1.  **Increased Accuracy and Robustness:**
    *   The model's ability to learn motion patterns and leverage short-term object history should lead to more accurate bounding box predictions, especially in cluttered environments or when UAVs are distant or move quickly.
    *   Improved differentiation from non-target moving objects (e.g., birds, other airborne debris).

2.  **Better Handling of Challenging Scenarios:**
    *   **Occlusions:** Transient partial occlusions are less likely to cause missed detections if the model can "remember" the UAV from previous frames.
    *   **Varying Appearances:** The model becomes more robust to rapid changes in UAV appearance due to perspective shifts or lighting variations.

3.  **Smoother, More Stable Detections:**
    *   Reduced "flicker" in detections across consecutive frames, leading to more reliable tracking and a better user experience if deployed in a real-time system.

4.  **Potential for Early Detection:**
    *   By analyzing initial motion cues, temporal models might offer the potential for earlier detection of approaching UAVs compared to models relying solely on static appearance in a single frame.

In summary, the temporal module enhances our UAV detection system by enabling it to learn from the dynamics of video sequences, promising more accurate, robust, and consistent performance, particularly in complex, real-world scenarios.

## Key Features & Recent Improvements