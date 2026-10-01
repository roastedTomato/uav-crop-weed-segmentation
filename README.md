# UAV Crop-Weed Segmentation

An in-progress RGB-only semantic segmentation project for identifying crop, weed and soil/background regions in UAV images of sorghum fields. Implemented in Python notebooks, with TensorFlow/Keras models planned for later stages.

## Current status

- Dataset snapshot acquisition and checksum verification implemented.
- Original image/mask pairing, dimensions and label colours inspected.
- Source-image splits and patch manifests prepared in earlier project scripts.
- The current notebook implements acquisition and file inspection; later sections are placeholders.
- Model training, CNN/U-Net comparison and final evaluation are **not yet completed**. No model accuracy or deployment performance is claimed.

Recorded source splits contain 9 training, 3 validation and 7 test images, plus 2 additional images from different growth stages. Splitting by image prevents reuse of the same source image across splits; spatial overlap between different images or flights remains unverified.

## Planned experiments

Compare a majority-class baseline, a small encoder-decoder CNN and U-Net. Report mean IoU, per-class IoU (especially weed IoU), Dice and visual error analysis. Ignore padded mask pixels labelled 255 in losses and metrics.

## Files

- `project_todo.ipynb`: current notebook entry point.
- `project_plan.md`: bilingual project definition, provenance and remaining work.
- `archive/audit_dataset.py`, `archive/prepare_splits.py`: earlier data audit and patch-generation scripts, retained for reproducibility.
- `artifacts/`: recorded audit results, source splits and patch manifest; no dataset images included.
- `requirements.txt`: dependencies for data preparation and planned training.

## Run locally

Use Python 3.11–3.13 for TensorFlow compatibility. Dependencies are not version-pinned; a fresh installation and the complete workflow have not been verified. The current local project environment does not have TensorFlow installed.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab project_todo.ipynb
```

On Windows, activate with `.venv\Scripts\activate`. Run from the project directory. Read notebook sections and `project_plan.md` before execution. The download is approximately 2.59 GB; extraction and patch generation need additional space. The code reuses a matching local archive or downloads it and validates a pinned SHA-256 snapshot. A changed dataset snapshot requires investigation before updating the checksum.

Local data goes under `data/` and is not committed. Recorded artifacts describe the original local preparation and are not proof of fresh execution after cloning. The current notebook does not yet regenerate every split/patch artifact; consult the earlier scripts and project plan.

## Dataset attribution

Original study: Genze et al. (2022), *Deep learning-based early weed segmentation using motion blurred UAV images of sorghum fields*, [paper DOI](https://doi.org/10.1016/j.compag.2022.107388), [original dataset v4](https://data.mendeley.com/datasets/4hh45vkp38/4), [authors' code](https://github.com/grimmlab/UAVWeedSegmentation).

This project uses [Bouhadjer's modified Kaggle dataset](https://www.kaggle.com/datasets/bouhadjer/crop-weed-segmentation-uav-rgb-indices), selecting original RGB images and masks and preparing source-image-based splits and patches. Refer to the dataset's bundled licence and the provenance discussion in `project_plan.md`. Original data and downloaded archives are excluded. The notebook retains small inspection visualisations from the local run; standalone overlay image files are excluded.

## Scope and limitations

This is an academic project in progress, not a deployed agricultural system. It does not perform species identification, geolocation, instance counting or automatic spraying. The images represent the dataset's sorghum-field setting; generalisation to other regions or crops has not been established.

AI assistance is part of the development workflow; the student must review, understand and verify submitted code. Repository preparation added this README and publication exclusions without changing the model implementation or claiming new experimental results.
