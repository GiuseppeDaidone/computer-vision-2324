import cv2
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Midas:
    """
        Face Depth Map estimation with MiDaS Monocular Depth Estimation

       :param input_dir: input directory of RGB image
       :param output_dir: output directory of image depth
       :param model_type: "DPT_Large" (MiDaS v3 - Large) | "DPT_Hybrid" (MiDaS v3 - Hybrid) | "MiDaS_small" (MiDaS v2.1 - Small)
    """

    def __init__(self, input_dir="", output_dir="", model_type="DPT_Large"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_type = model_type

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def depth_map(self, filename, save=True, output=False):
        """
            Compute depth map of an RGB image

            :param filename: name of the file
            :param save: flag to save depth map
            :param output: flag to show the depth map as output
            :return: depth map tensor
        """
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            """
                For Apple silicon processors (M-):
                    device = torch.device("mps") if torch.mps else torch.device("cpu")
                Not fully supported by pytorch
            """
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            print(f"Processing {filename} with MiDaS on {device}")

            midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
            midas.to(device)
            midas.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
                transform = midas_transforms.dpt_transform
            else:
                transform = midas_transforms.small_transform

            img = cv2.imread(self.input_dir + "/" + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            input_batch = transform(img).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Process depth map
            depth_map = prediction.cpu().numpy()
            depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)
            depth_map = (depth_map * 65535 / np.max(depth_map)).astype(np.uint16)

            if save:
                # Save depth map
                depth_map_path = os.path.join(self.output_dir, f"{filename}_depth.png")
                cv2.imwrite(depth_map_path, depth_map)
                print(f"Depth map saved at {depth_map_path}")

            if output:
                # Visualize output
                print(f"Printing Depth Map of {filename}")
                image, axis = plt.subplots(1, 2)
                axis[0].axis('off')
                axis[0].imshow(img)
                axis[1].axis('off')
                axis[1].imshow(depth_map, cmap='jet')
                plt.suptitle(f"RGB and Depth Map of {filename}")
                plt.show()
            return torch.tensor(depth_map)


# Example
#test = Midas("faceforensics_frames", "faceforensics_depth_maps", "DPT_Hybrid")
#for filename in os.listdir("faceforensics_frames"):
#    test.depth_map(filename, output=True)
