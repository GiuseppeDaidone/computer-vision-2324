# Layer-wise Depth Integration in RGBD Deepfake Detection
Repository for Computer Vision Project a.y. 2023/2024 - Sapienza University of Rome

## Repository Organization
* src : source code directory
  * _data_augmentation.py_ : class which can apply data augmentation operations (flip horiz, flip vert, rotation, gaussian noise, salt-and-pepper noise)
  * _deepfake_detector_input_d.py_ : deepfake detector with RGBD as input layer
  * _deepfake_detector_mid_d.py_ : deepfake detector with RGB as input layer, D as mid layer
  * _deepfake_detector_output_d.py_ : deepfake detector with RGB as input layer, D as output layer
  * _face_depth_net.py_ : face depth map estimation class via [MiDaS](https://github.com/isl-org/MiDaS)
  * _face_extractor.py_ : face extractor class using [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector)
  * _faceforensics_download_v4.py_ : code and terminal commands as comments to download FaceForensics++ dataset (for more information to download the dataset, please consult [official repository](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md))
  * _frame_extractor.py_ : class which extract video frames and save them as RGB images
  * _preprocessor.py_ : code to compute all the preprocessing operations pipeline
  * _deepfake_detector_input_d.pth_ : PyTorch model for deepfake detector with input depth
  * _deepfake_detector_mid_d.pth_ : PyTorch model for deepfake detector with mid depth
  * _deepfake_detector_output_d.pth_ : PyTorch model for deepfake detector with output depth
* plots : plots displayed in the presentation
* Computer Vision Project Presentation : presentation slides containing all the details on the architecture and the conducted study

## Project Credits
Giuseppe Daidone 2122594, MSc Engineering in Computer Science
