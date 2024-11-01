import cv2
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch

model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
proc = VideoMAEImageProcessor.from_pretrained(model_name)
model = VideoMAEForVideoClassification.from_pretrained(model_name)

# get video
video_path = "test.mov"
video = cv2.VideoCapture(video_path)

fps = int(video.get(cv2.CAP_PROP_FPS))
num_frames = 16
f_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
f_count = 0
results = []

# Sample frames evenly throughout the video
# I dont understand what this means? frame evenly? what?
f_indices = torch.linspace(0, f_frame - 1, steps=num_frames).long()
for idx in range(f_frame):
    success, frame = video.read()
    if not success:
        break

    if idx in f_indices:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results.append(frame_rgb)
video.release()

inputs = proc(results, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_idx = logits.argmax(-1).item()
    predicted = model.config.id2label[predicted_idx]

print("Video predicted as:", predicted)