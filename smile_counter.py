import cv2

class SmileCounter:
    def __init__(self, video_path):
        self.video_path = video_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.cap = cv2.VideoCapture(self.video_path)
        self.smile_count = 0
        self.smile_detected_in_frame = False

    def process_video(self):
        if not self.cap.isOpened():
            print("Не удалось открыть видео.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                smiles = self.smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
                if len(smiles) > 0 and not self.smile_detected_in_frame:
                    self.smile_count += 1
                    self.smile_detected_in_frame = True
                elif len(smiles) == 0:
                    self.smile_detected_in_frame = False
        self.cap.release()
        return self.smile_count
