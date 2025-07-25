import cv2
import numpy as np
import threading
from time import time
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from collections import deque

class MotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Amplifier 1.0")
        self.root.geometry("400x400")
        self.root.resizable(False, False)
        style = ttk.Style()
        style.theme_use("vista")

        self.running = False
        self.capture = None
        self.video_source = "Camera"
        self.thread = None
        self.frame_stack = deque()
        self.fps = 60
        self.previous_time = time()
        self.video_file_path = None
        self.create_widgets()

    def create_widgets(self):
        main = Frame(self.root, padx=10, pady=10)
        main.pack(fill=BOTH, expand=True)

        self.source_var = StringVar(value="Camera")
        ttk.Label(main, text="Source:").pack(anchor=W)
        source_menu = ttk.Combobox(main, textvariable=self.source_var, values=["Camera", "Video File"], state="readonly")
        source_menu.pack(fill=X)
        source_menu.bind("<<ComboboxSelected>>", self.source_changed)

        self.browse_button = ttk.Button(main, text="Browse...", command=self.browse_file)
        self.browse_button.pack(fill=X)
        self.browse_button.configure(state=DISABLED)

        self.threshold = self.create_slider(main, "Threshold", 1, 100, 10)
        self.amplification = self.create_slider(main, "Amplification", 1, 30, 10)
        self.blur = self.create_slider(main, "Gaussian Blur", 0, 15, 3)
        self.stack_size = self.create_slider(main, "Stack Size", 2, 30, 5)

        self.start_button = ttk.Button(main, text="Start", command=self.toggle)
        self.start_button.pack(pady=(10, 5), fill=X)

        self.status = StringVar(value="Idle")
        ttk.Label(main, textvariable=self.status, anchor=CENTER, relief=SUNKEN).pack(fill=X, pady=(10, 0))

    def create_slider(self, parent, label, minval, maxval, init):
        ttk.Label(parent, text=label).pack(anchor=W)
        var = IntVar(value=init)
        scale = ttk.Scale(parent, from_=minval, to=maxval, variable=var, orient=HORIZONTAL)
        scale.pack(fill=X)
        return var

    def source_changed(self, event=None):
        if self.source_var.get() == "Video File":
            self.browse_button.configure(state=NORMAL)
        else:
            self.browse_button.configure(state=DISABLED)
            self.video_file_path = None

    def browse_file(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_file_path = file_path

    def toggle(self):
        if self.running:
            self.running = False
            self.start_button.config(text="Start")
            self.status.set("Stopping...")
        else:
            self.running = True
            self.start_button.config(text="Stop")
            self.status.set("Running")
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def run(self):
        src = 0 if self.source_var.get() == "Camera" else self.video_file_path
        self.capture = cv2.VideoCapture(src)

        if not self.capture.isOpened():
            messagebox.showerror("Error", "Could not open video source.")
            self.status.set("Idle")
            self.running = False
            return

        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.frame_stack = deque(maxlen=self.stack_size.get())

        ret, frame = self.capture.read()
        if not ret:
            self.status.set("Failed to read video.")
            self.running = False
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur.get() > 0:
            gray = cv2.GaussianBlur(gray, (self.blur.get() | 1, self.blur.get() | 1), 0)

        self.frame_stack.append(gray)

        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                break

            now = time()
            delta = now - self.previous_time
            self.previous_time = now
            self.fps = self.fps * 0.9 + (1.0 / delta) * 0.1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.blur.get() > 0:
                gray = cv2.GaussianBlur(gray, (self.blur.get() | 1, self.blur.get() | 1), 0)

            self.frame_stack.append(gray)

            diff = np.zeros_like(gray, dtype=np.float32)
            for past in self.frame_stack:
                diff += cv2.absdiff(gray, past).astype(np.float32)
            diff /= len(self.frame_stack)

            amplified = cv2.convertScaleAbs(diff * self.amplification.get())
            thresh_value = max(self.threshold.get(), np.mean(diff) * 1.5)
            _, mask = cv2.threshold(amplified, thresh_value, 255, cv2.THRESH_BINARY)
            motion = cv2.bitwise_and(amplified, amplified, mask=mask)
            output = cv2.merge([motion]*3)

            cv2.putText(output, f"{self.fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Motion Amplifier", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.status.set("Idle")
        self.start_button.config(text="Start")
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = Tk()
    app = MotionApp(root)
    root.mainloop()
