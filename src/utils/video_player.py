'''
This code was generated and revised by ChatGPT
'''

import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os

class VideoPlayer:
    def __init__(self, root, video_path):
        self.root = root
        self.cap = cv2.VideoCapture(video_path)
        self.paused = True
        self.current_frame_idx = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.saved_timestamps = []
        self.pen_mode = False
        self.drawing = False
        self.video_path = video_path

        # Read first frame to get dimensions
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read video")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_rgb = frame.copy()
        self.height, self.width = frame.shape[:2]
        self.global_overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # GUI setup
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left side: video + controls
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT)

        self.canvas = tk.Label(self.left_frame)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        self.timestamp_label = tk.Label(self.left_frame, text="00:00.000 / 00:00.000")
        self.timestamp_label.pack()

        controls = tk.Frame(self.left_frame)
        controls.pack()

        tk.Button(controls, text="Play/Pause", command=self.toggle_play).pack(side=tk.LEFT)
        tk.Button(controls, text="Prev", command=self.prev_frame).pack(side=tk.LEFT)
        tk.Button(controls, text="Next", command=self.next_frame).pack(side=tk.LEFT)
        tk.Button(controls, text="Save Time", command=self.save_timestamp).pack(side=tk.LEFT)
        tk.Button(controls, text="Delete Time", command=self.delete_timestamp).pack(side=tk.LEFT)
        tk.Button(controls, text="Pen On/Off", command=self.toggle_pen).pack(side=tk.LEFT)
        tk.Button(controls, text="Export", command=self.export_results).pack(side=tk.LEFT)

        # Right side: timestamp list
        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(self.right_frame, text="Saved Timestamps").pack()
        self.timestamp_listbox = tk.Listbox(self.right_frame, width=20)
        self.timestamp_listbox.pack(padx=10, pady=5)

        self.update_frame()

    def toggle_play(self):
        self.paused = not self.paused

    def toggle_pen(self):
        self.pen_mode = not self.pen_mode

    def format_time(self, frame_idx):
        time_in_seconds = frame_idx / self.fps
        minutes = int(time_in_seconds // 60)
        seconds = int(time_in_seconds % 60)
        milliseconds = int((time_in_seconds - int(time_in_seconds)) * 1000)
        return f"{minutes:02}:{seconds:02}.{milliseconds:03}"

    def update_frame(self):
        if not self.paused:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_idx = 0
                ret, frame = self.cap.read()
            else:
                self.current_frame_idx += 1
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_rgb = frame.copy()

            # Apply global overlay
            frame = cv2.addWeighted(frame, 1.0, self.global_overlay, 1.0, 0)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(img)
            self.canvas.imgtk = imgtk
            self.canvas.config(image=imgtk)

        current = self.format_time(self.current_frame_idx)
        total = self.format_time(self.total_frames)
        self.timestamp_label.config(text=f"{current} / {total}")
        self.root.after(30, self.update_frame)

    def next_frame(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.paused = True

    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.paused = True

    def save_timestamp(self):
        timestamp = self.format_time(self.current_frame_idx)
        if timestamp not in self.saved_timestamps:
            self.saved_timestamps.append(timestamp)
            self.timestamp_listbox.insert(tk.END, timestamp)

    def delete_timestamp(self):
        selected = self.timestamp_listbox.curselection()
        if selected:
            idx = selected[0]
            value = self.timestamp_listbox.get(idx)
            self.timestamp_listbox.delete(idx)
            self.saved_timestamps.remove(value)

    def start_draw(self, event):
        if not self.pen_mode:
            return
        self.drawing = True
        self.prev_x, self.prev_y = event.x, event.y

    def draw(self, event):
        if self.drawing and self.pen_mode:
            x1, y1 = self.prev_x, self.prev_y
            x2, y2 = event.x, event.y
            cv2.line(self.global_overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            self.prev_x, self.prev_y = x2, y2

    def stop_draw(self, event):
        self.drawing = False

    def export_results(self):
        folder = filedialog.askdirectory(title="Select Export Folder")
        if not folder:
            return

        # Save timestamps
        ts_path = os.path.join(folder, "timestamps.txt")
        with open(ts_path, "w") as f:
            for t in self.saved_timestamps:
                f.write(f"{t}\n")

        # Save frame with global overlay
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_idx in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            drawn = cv2.addWeighted(frame, 1.0, self.global_overlay, 1.0, 0)
            out_path = os.path.join(folder, f"frame_{frame_idx:04}.png")
            cv2.imwrite(out_path, cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))

        print(f"Exported to: {folder}")


# Run the video player
if __name__ == "__main__":
    # run this file from root directory
    #todo adjust this part so that arguments can be given from command line
    video_path = "data/input/test2.mp4"
    video_title = "our beautiful video player!"

    root = tk.Tk()
    root.title(video_title)
    player = VideoPlayer(root, video_path=video_path)  # Replace with your video path
    root.mainloop()