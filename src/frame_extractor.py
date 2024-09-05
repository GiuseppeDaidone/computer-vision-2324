import cv2
import os


class VideoFrameExtractor:
    """
        Video Frame Extractor

        :param input_dir: input directory of video
        :param output_dir: output directory of video frames
        :param save_rate: express rate of frame saving (ex: 1 -> 1 f/s, 2 -> 0.5 f/s, 0.5 -> 2 f/s)
    """

    def __init__(self, input_dir, output_dir, save_rate=1):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.save_rate = save_rate

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_frames(self, filename):
        """
            Extract frames from video
            :param filename: name of the file
        """
        self.filename = filename
        self.video_cap = cv2.VideoCapture(self.input_dir + "/" + filename)
        self.fps = int(self.video_cap.get(cv2.CAP_PROP_FPS))

        current_frame = 0
        save_interval = self.fps * self.save_rate

        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret:
                break

            if current_frame % save_interval == 0:
                frame_time = current_frame / self.fps
                self.save_frame(frame, frame_time)

            current_frame += 1

        self.video_cap.release()
        cv2.destroyAllWindows()

    def save_frame(self, frame, frame_time):
        frame_filename = f"{self.filename}_frame_{frame_time}.png"
        frame_path = os.path.join(self.output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        print(f"Saved frame at {frame_time:.2f} seconds as {frame_filename}")

# Example
#extractor = VideoFrameExtractor("tmp_vid", "faceforensics_frames", save_rate=1)
#extractor.extract_frames("01__exit_phone_room.mp4")
