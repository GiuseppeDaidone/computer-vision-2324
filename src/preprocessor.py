from frame_extractor import VideoFrameExtractor
from data_augmentation import DataAugmentation
from face_extractor import FaceExtractor
from face_depth_net import Midas
import os
import random

"""
    Preprocessing operations in pipeline
"""

original_sequences = "original_sequences/"
youtube = "youtube/c23/videos"
manipulated_sequences = "manipulated_sequences/"
deepfakes = "Deepfakes/c23/videos"

video_dir = "faceforensics_dataset/" + manipulated_sequences + deepfakes
frames_dir = "faceforensics_frames/" + manipulated_sequences + deepfakes
faces_dir = "faceforensics_faces/" + manipulated_sequences + deepfakes
depth_maps_dir = "faceforensics_depth_maps/" + manipulated_sequences + deepfakes

# Extract frames from video
frames = VideoFrameExtractor(video_dir, frames_dir, save_rate=3)  # Extract frame every 3 seconds
for filename in os.listdir(video_dir):
    frames.extract_frames(filename)

# Augment data
aug = DataAugmentation(frames_dir, frames_dir)
for filename in os.listdir(frames_dir):
    if random.random() < 0.3:  # Flip vertically with 30% of probability
        aug.flip_vertically(filename)
    if random.random() < 0.2:  # Random rotation with 20% of probability
        aug.rotate(filename)
    if random.random() < 0.1:  # Add Gaussian noise with 10% of probability
        aug.gaussian_noise(filename)
    if random.random() < 0.1:
        aug.salt_pepper_noise(filename)  # Add salt-and-pepper noise with 10% of probability

# Extract faces from frames
faces = FaceExtractor(input_dir=frames_dir, output_dir=faces_dir, window_size=224)
for filename in os.listdir(frames_dir):
    faces.extract_face(filename, extract_once=True)

# Compute depth map from faces
depth = Midas(faces_dir, depth_maps_dir, "DPT_Hybrid")
for filename in os.listdir(faces_dir):
    depth.depth_map(filename, output=False)

