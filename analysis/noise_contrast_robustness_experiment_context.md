# Noise / Contrast Robustness Experiment Context

## Goal
Evaluate whether segmentation performance is maintained when noise and contrast are perturbed on a held-out test set that already has ground-truth labels.

## Recommended model checkpoint
Use this trained YOLO instance segmentation checkpoint:

`C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\Model Training by Yifei\round_2\results_yolov8_heavy_augmentation\training\vesicle_instance_seg_v2\weights\best.pt`

Other available checkpoints in the same run:
- `epoch0.pt`
- `epoch50.pt`
- `epoch100.pt`
- `epoch150.pt`
- `last.pt`

## Dataset root
Held-out dataset root:

`C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\Model Training by Yifei\round_2\results_yolov8_heavy_augmentation\dataset`

## Test split to use
Images:

`C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\Model Training by Yifei\round_2\results_yolov8_heavy_augmentation\dataset\images\test`

Labels:

`C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\Model Training by Yifei\round_2\results_yolov8_heavy_augmentation\dataset\labels\test`

Verified counts on 2026-05-06:
- 12 test images
- 12 test label files

Example test files:
- `0005_Jul30.jpg`
- `0008_Jul30.jpg`
- `3_square247_hole45_0_hm.jpg`

## Dataset YAML
Current dataset YAML file:

`C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\Model Training by Yifei\round_2\results_yolov8_heavy_augmentation\dataset\dataset.yaml`

Important caveat:
- `dataset.yaml` still contains an old absolute path under `C:\Users\Yifei\...`
- For any new evaluation script, either rewrite `path:` at runtime or ignore it and pass local image/label paths directly.

Current contents:
- `train: images/train`
- `val: images/val`
- `names[0]: vesicle`
- `nc: 1`

## Prior related results in this workspace
Recent standard EV detection + morphology run was done on a separate 9-image folder and produced:
- YOLO polygon labels
- segmentation boundaries JSON
- morphology CSV outputs
- ellipse-fit comparison outputs

This is separate from the robustness experiment. For the new experiment, use the held-out 12-image test split above because it already has ground-truth labels.

## Suggested experiment framing
Primary question:
- Does segmentation quality remain stable under controlled degradation of noise and contrast?

Suggested perturbation axes:
- additive Gaussian noise
- salt-and-pepper or impulse noise
- contrast scaling
- brightness shift
- optional blur

Suggested evaluation outputs:
- per-image and aggregate segmentation metrics against ground truth
- at minimum: mask IoU, precision, recall, F1
- save qualitative overlays for each perturbation level
- preserve original test set as baseline control

## Practical implementation notes
- The repo's `inference/inference.py` entrypoint is hardcoded for older paths, so a dedicated evaluation script is cleaner than editing that config repeatedly.
- The repo already has YOLO polygon label handling utilities in `annotation/io.py`.
- GPU inference is working after fixing CUDA mask-to-numpy conversion in `inference/inference.py`.

## Good starting ask for a new thread
"Run a robustness experiment on the held-out test split with existing ground-truth labels. Use the checkpoint at `C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\Model Training by Yifei\round_2\results_yolov8_heavy_augmentation\training\vesicle_instance_seg_v2\weights\best.pt` and the test set under `C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\Model Training by Yifei\round_2\results_yolov8_heavy_augmentation\dataset\images\test` and `...\labels\test`. Create a script that perturbs noise and contrast at several levels, runs segmentation, compares predictions against ground truth, and saves both summary metrics and qualitative overlays. Be careful that `dataset.yaml` still points to an old `C:\Users\Yifei\...` path, so use the local ML-2619 paths instead."