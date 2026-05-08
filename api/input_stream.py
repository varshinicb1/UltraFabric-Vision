import cv2

class VideoStream:
    def __init__(self, source=0):
        """
        source: int for webcam, str for RTSP or video file
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            # If it's a video file, loop it
            if isinstance(self.source, str) and not self.source.startswith("rtsp"):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            else:
                return False, None
        return ret, frame

    def release(self):
        self.cap.release()
