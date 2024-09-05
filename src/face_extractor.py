import cv2
import mediapipe as mp
import os
import numpy as np


class FaceExtractor:
    """
        Face Extractor using MediaPipe

        :param input_dir: directory containing input images
        :param output_dir: directory to save extracted faces
        :param window_size: size of the square window to resize or pad extracted faces (if needed)
    """

    def __init__(self, input_dir="", output_dir="", window_size=224):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.window_size = window_size

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_face(self, filename, extract_once=False):
        """
        Extract face(s) from an image

        :param filename: name of the file
        :param extract_once: if True, stops after extracting the first detected face
        """
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            print(f"Extracting face from {filename}")

            image_path = os.path.join(self.input_dir, filename)
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Perform face detection
            results = self.face_detection.process(img_rgb)

            # Check if at least one face was found
            if results.detections:
                # Loop over the detected faces
                for i, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = img.shape
                    x, y, w, h = (
                        int(bboxC.xmin * w),
                        int(bboxC.ymin * h),
                        int(bboxC.width * w),
                        int(bboxC.height * h),
                    )

                    # Calculate the center of the face
                    cx, cy = x + w // 2, y + h // 2

                    # Calculate the square region around the face's center
                    half_size = self.window_size // 2
                    x1, y1 = max(cx - half_size, 0), max(cy - half_size, 0)
                    x2, y2 = min(cx + half_size, img.shape[1]), min(cy + half_size, img.shape[0])

                    # Extract the face region
                    face = img[y1:y2, x1:x2]

                    # If the extracted face is not the correct size, pad it
                    if face.shape[0] != self.window_size or face.shape[1] != self.window_size:
                        face = cv2.copyMakeBorder(face,
                                                  top=max(0, half_size - (cy - y1)),
                                                  bottom=max(0, half_size - (y2 - cy)),
                                                  left=max(0, half_size - (cx - x1)),
                                                  right=max(0, half_size - (x2 - cx)),
                                                  borderType=cv2.BORDER_CONSTANT,
                                                  value=[0, 0, 0])

                    # Ensure the face is exactly the window size
                    face = cv2.resize(face, (self.window_size, self.window_size))

                    # Save the extracted face to the output path
                    output_path = os.path.join(self.output_dir, f"{filename}_face{i}.png")
                    cv2.imwrite(output_path, face)

                    print(f"Face {i + 1} extracted and saved to {output_path}")

                    if extract_once:
                        break
            else:
                print(f"No face found in {filename}")

# Example
# extractor = FaceExtractor(input_dir="tmp_img", output_dir="faceforensics_faces", window_size=224)
# extractor.extract_face("47339-3678227892.jpg", extract_once=True)
