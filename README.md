# Bone-Segnet
Bone-Segnet is a network for segmenting hypermetabolic bone lesions in whole-body SPECT. It addresses low contrast, small lesions, and complex distributions, enabling accurate lesion segmentation and subsequent quantitative analysis for objective characterization of lesion patterns.

## Network Training
python trainnet.py

During the training stage, anterior and posterior whole-body SPECT bone scintigraphy images and their corresponding pixel-level annotation masks are first loaded. The images are then preprocessed, including resizing, intensity normalization, and data augmentation.
The processed data are fed into the Bone-Segnet network for training. Mini-batch data loading is adopted, and mixed precision training is utilized to improve computational efficiency and reduce GPU memory consumption.
After training, the model weights are saved for subsequent testing and inference.

## Network Inference
python predictnet.py

During the inference stage, the trained model weights are loaded, and the input whole-body SPECT bone scintigraphy images are preprocessed and fed into the network to generate segmentation predictions.

The resulting segmentation outputs can be further used for quantitative analysis, including lesion count, pixel burden, and distribution.
