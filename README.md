# UAV Crop-Weed Segmentation

RGB semantic segmentation of **soil/background, sorghum crops and weeds** in UAV imagery, using Python and TensorFlow/Keras. This completed academic project covers image-mask validation, source-image splitting, patch generation, model training and held-out test evaluation.

A majority-class baseline, a small encoder-decoder CNN and U-Net are compared on the same task. **U-Net achieved a test Mean IoU of 0.8156 and Weed IoU of 0.6627**, outperforming the smaller CNN on this dataset split.

## Results

### Held-out test set

Evaluation uses **7 source images / 2,310 patches**. Scores are calculated from accumulated confusion matrices over valid pixels; padding and ignored pixels are excluded. Mean IoU and Mean Dice are unweighted averages across the three classes.

| Model | Mean IoU | Mean Dice | Soil IoU | Crop IoU | Weed IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.3290 | 0.3312 | 0.9870 | 0.0000 | 0.0000 |
| Small encoder-decoder CNN | 0.7827 | 0.8687 | 0.9975 | 0.7533 | 0.5975 |
| **U-Net** | **0.8156** | **0.8922** | 0.9974 | **0.7867** | **0.6627** |

U-Net improves Mean IoU by **3.29 percentage points** and Weed IoU by **6.53 percentage points** over the small CNN. The majority baseline predicts soil everywhere: its high soil IoU reflects the strong class imbalance, while its zero crop and weed IoU exposes its limitations.

### Validation and generalisation

| Model | Validation Mean IoU | Validation Weed IoU | Test Mean IoU | Test Weed IoU |
| --- | ---: | ---: | ---: | ---: |
| Small CNN | 0.8374 | 0.7221 | 0.7827 | 0.5975 |
| U-Net | 0.8502 | 0.7449 | 0.8156 | 0.6627 |

Checkpoints are selected by validation loss. The test set is reserved for final evaluation, rather than tuning. Both networks score lower on the test set than on validation, showing a generalisation gap.

**Explore the recorded outputs:**

- [Final test evaluation](results/06_test_evaluation.html): metrics, comparison charts and RGB / ground truth / CNN / U-Net prediction panels.
- [Validation comparison](results/05_evaluation_comparison.html).
- [CNN training and evaluation](results/03_small_encoder_decoder.html).
- [U-Net training and evaluation](results/04_unet.html).
- [Data preparation](results/01_data_preparation.html) and [majority baseline](results/02_majority_baseline.html).

Download the HTML files and open them in a browser to view the exported notebooks. Prediction examples select the patch with the **most labelled weed pixels per test image**; they illustrate weed segmentation and are not a random sample of test patches.

## Data and preprocessing

The project uses original RGB images and segmentation masks from [Crop-Weed Segmentation UAV RGB+Indices on Kaggle](https://www.kaggle.com/datasets/bouhadjer/crop-weed-segmentation-uav-rgb-indices). Only RGB inputs are used; vegetation indices are not model inputs.

| Class ID | Class | Mask RGB colour |
| --- | --- | --- |
| 0 | Background / soil | `(199, 199, 199)` |
| 1 | Crop / sorghum | `(31, 119, 180)` |
| 2 | Weeds | `(255, 127, 14)` |

- Validate image-mask identifiers, dimensions and expected mask colours.
- Split the 12 train/validation source images into **9 training and 3 validation images**, using seed **42**, before extracting patches.
- Use the separate **7-image test set** for final evaluation. Two extra growth-stage images appear in the earlier audit but are not part of the reported main test results.
- Extract non-overlapping **256 × 256** patches and normalise RGB values to `[0, 1]`.
- Pad edge patches and use zero pixel weights for padding / ignored regions (`255` before safe-label replacement).

The recorded audit lists 2,970 training, 990 validation and 2,310 test patches. Splitting at source-image level prevents patches from the same original appearing in different splits. Flight identity and spatial overlap between different original images remain unverified.

## Models and training

| Model | Design |
| --- | --- |
| Majority baseline | Predict the most frequent valid training-pixel class, without using image content. |
| Small encoder-decoder CNN | Two pooling stages, 16/32-channel encoder blocks, a 64-channel bottleneck and a decoder with skip connections. |
| U-Net | Three pooling stages, 16/32/64-channel encoder blocks, a 128-channel bottleneck and three decoder stages with skip connections. |

Both neural networks use ReLU convolution blocks, bottleneck spatial dropout (`0.2`) and a final 1 × 1 convolution with three-class softmax. Training uses Adam (`learning rate = 0.001`), sparse categorical cross-entropy with pixel weights, batch size **8**, up to **30 epochs**, and early stopping with patience **5**. The best checkpoint is saved according to validation loss.

## Run locally

The recorded experiment environment used **Python 3.12.14, TensorFlow 2.16.2, Keras 3.15.1 and NumPy 1.26.4** on macOS ARM64. `requirements.txt` is not version-pinned, so a fresh installation can differ from that environment.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

On Windows, activate with `.venv\Scripts\activate`. Launch Jupyter from the repository root.

### Prepare the dataset

Download and extract the dataset linked above. The current notebook expects local files in this layout and **does not download the dataset automatically**:

```text
data/raw/
├── trainval/
│   ├── img/*.jpg
│   └── msk/*.png
└── test/
    ├── img/*.jpg
    └── msk/*.png
```

Each main image and its mask must share the same filename stem. The dataset download is approximately 2.59 GB, with additional space needed for extraction and generated outputs. Dataset files and trained checkpoints are excluded from Git.

### Execute the notebooks in order

| Notebook | Purpose |
| --- | --- |
| `01_data_preparation.ipynb` | Validate images/masks, generate `artifacts/split.json` and inspect patch datasets. |
| `02_majority_baseline.ipynb` | Determine the majority class from training masks and evaluate the validation baseline. |
| `03_small_encoder_decoder.ipynb` | Train the small CNN and save its best checkpoint, history and validation scores. |
| `04_unet.ipynb` | Train U-Net and save its best checkpoint, history and validation scores. |
| `05_evaluation_comparison.ipynb` | Compare the validation results. |
| `06_test_evaluation.ipynb` | Evaluate the saved checkpoints on test images and generate prediction examples. |

Run every cell in each notebook in order. The CNN notebook explicitly selects CPU and should start in a fresh kernel. U-Net training and test evaluation materialise patches in memory, so sufficient RAM is required. Retrain both models before reproducing the final comparison: their `.keras` checkpoints are not included in the repository.

The README summarises the existing recorded experiments; model training was not rerun during this documentation update. Random seeds improve repeatability but do not guarantee identical results across platforms and dependency versions.

## Repository layout

```text
01_data_preparation.ipynb
02_majority_baseline.ipynb
03_small_encoder_decoder.ipynb
04_unet.ipynb
05_evaluation_comparison.ipynb
06_test_evaluation.ipynb
results/             Exported notebooks and supporting reports
artifacts/           Published audit metadata; local runs also generate scores and checkpoints
archive/             Earlier dataset audit and split-preparation scripts
project_plan.md      Earlier planning and provenance notes
requirements.txt     Python dependencies
```

The numbered notebooks and recorded results describe the completed workflow. Some planning notes retain the earlier development status.

## Limitations

- The dataset contains few original images and is strongly dominated by soil pixels. Patch counts should not be interpreted as independent field observations.
- Results describe one fixed split, without repeated-seed uncertainty estimates or verification of flight-level separation.
- Generalisation to other crops, regions, cameras and field conditions has not been established.
- This is a segmentation experiment, with no deployed spraying system, species identification, geolocation or real-time performance claim.

## Dataset attribution

Original study: Genze et al. (2022), *Deep learning-based early weed segmentation using motion blurred UAV images of sorghum fields*: [paper](https://doi.org/10.1016/j.compag.2022.107388), [original dataset v4](https://data.mendeley.com/datasets/4hh45vkp38/4), [authors' code](https://github.com/grimmlab/UAVWeedSegmentation).

This project uses Bouhadjer's modified Kaggle distribution. Consult the dataset's bundled licence and the provenance notes in `project_plan.md` before reusing the data. Original images and downloaded archives are not redistributed here.

AI tools assisted development. The reported metrics come from the saved project experiments, rather than AI-generated estimates.
