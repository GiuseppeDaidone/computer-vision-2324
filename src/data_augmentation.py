import numpy as np
import cv2
import random
import os


class DataAugmentation:
    """
        Data augmentation operations
        :param input_dir: directory containing input image
        :param output_dir: directory to save augmented image
    """
    def __init__(self, input_dir="", output_dir=""):
        self.input_dir = input_dir
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def flip_horizontally(self, filename):
        """
            Flips the image horizontally

            :param filename: input image
            :return: horizontally flipped image
        """
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            print(f"Flipping horizontally {filename}")

            # Open image
            image_path = os.path.join(self.input_dir, filename)
            img = cv2.imread(image_path)

            flip = cv2.flip(img, 1)

            # Save the flipped image
            output_path = os.path.join(self.output_dir, f"{filename}_flip_horiz.png")
            cv2.imwrite(output_path, flip)
            print(f"Horizontally flipped image saved to {output_path}")

            return flip

    def flip_vertically(self, filename):
        """
            Flips the image vertically

            :param filename: input image
            :return: vertically flipped image
        """
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            print(f"Flipping vertically {filename}")

            # Open image
            image_path = os.path.join(self.input_dir, filename)
            img = cv2.imread(image_path)

            flip = cv2.flip(img, 0)

            # Save the flipped image
            output_path = os.path.join(self.output_dir, f"{filename}_flip_vert.png")
            cv2.imwrite(output_path, flip)
            print(f"Vertically flipped image saved to {output_path}")

            return flip

    def rotate(self, filename, angle=None):
        """
            Rotates the image by the specified angle. If no angle is provided, rotates by a random angle

            :param filename: input image
            :param angle: angle by which to rotate the image (if None, a random angle is generated)
            :return: rotated image
        """
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            if angle is None:
                angle = random.uniform(0, 360)

            print(f"Rotating {filename} by an angle of {angle}")

            # Open image
            image_path = os.path.join(self.input_dir, filename)
            img = cv2.imread(image_path)

            # Get image dimensions
            (h, w) = img.shape[:2]
            # Get the center of the image
            center = (w // 2, h // 2)

            # Perform the rotation
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))

            # Save the rotated image
            output_path = os.path.join(self.output_dir, f"{filename}_rot.png")
            cv2.imwrite(output_path, rotated)
            print(f"Vertically flipped image saved to {output_path}")

            return rotated

    def gaussian_noise(self, filename, mean=0, std=25):
        """
            Adds Gaussian noise to the image

            :param filename: input image
            :param mean: the mean of the Gaussian noise
            :param std: the standard deviation of the Gaussian noise
            :return: image with Gaussian noise
        """
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            print(f"Adding Gaussian noise to {filename}")

            # Open image
            image_path = os.path.join(self.input_dir, filename)
            img = cv2.imread(image_path)

            gaussian = np.random.normal(mean, std, img.shape)
            noisy_image = img + gaussian

            # Clip the values to be in the valid range for an image [0, 255] and convert to uint8
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

            # Save the Gaussian noised image
            output_path = os.path.join(self.output_dir, f"{filename}_gauss_noise.png")
            cv2.imwrite(output_path, noisy_image)
            print(f"Gaussian noised image saved to {output_path}")

            return noisy_image

    def salt_pepper_noise(self, filename, salt_prob=0.01, pepper_prob=0.01):
        """
            Adds salt-and-pepper noise to the image

            :param filename: input image
            :param salt_prob: probability of adding salt (white pixels)
            :param pepper_prob: probability of adding pepper (black pixels)
            :return: image with added salt-and-pepper noise
        """
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            print(f"Adding salt-and-pepper noise to {filename}")

            # Open image
            image_path = os.path.join(self.input_dir, filename)
            img = cv2.imread(image_path)

            noisy_image = np.copy(img)

            # Salt noise (white pixels)
            salt_mask = np.random.rand(*img.shape[:2]) < salt_prob
            noisy_image[salt_mask] = 255

            # Pepper noise (black pixels)
            pepper_mask = np.random.rand(*img.shape[:2]) < pepper_prob
            noisy_image[pepper_mask] = 0

            # Save the Gaussian noised image
            output_path = os.path.join(self.output_dir, f"{filename}_sp_noise.png")
            cv2.imwrite(output_path, noisy_image)
            print(f"Salt-and-pepper noised image saved to {output_path}")

            return noisy_image


# Example
aug = DataAugmentation("tmp_img", "tmp_img")
#aug.flip_horizontally("man.jpg")
#aug.flip_vertically("man.jpg")
#aug.rotate("man.jpg")
#aug.gaussian_noise("man.jpg")
#aug.salt_pepper_noise("man.jpg")