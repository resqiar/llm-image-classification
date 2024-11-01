import cv2
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

model_name = "google/vit-base-patch16-224"
proc = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# get video
video_path = "test.mov"
output_path = "result_with_label.mp4"
video = cv2.VideoCapture(video_path)

fps = int(video.get(cv2.CAP_PROP_FPS))
f_count = 0
f_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
f_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
results = []

# video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (f_width, f_height))
pause_duration = 2  # seconds
pause_frames = int(pause_duration * fps) # if fps is 30 then 2s pause is 60 frames

print("start reading video")

# read the video every 1 sec
while video.isOpened():
    video.set(cv2.CAP_PROP_POS_FRAMES, f_count * fps)

    # read next frame
    success, frame = video.read()
    if not success:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    inputs = proc(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_idx = logits.argmax(-1).item()
        predicted = model.config.id2label[predicted_idx]

        print("success:", predicted)

        results.append(predicted)
    
    # Overlay the prediction label on the frame
    label_text = f"Frame Prediction: {predicted}"
    label_font = cv2.FONT_HERSHEY_COMPLEX
    label_scale = 2
    label_thickness = 3

    (text_width, text_height), baseline = cv2.getTextSize(label_text, label_font, label_scale, label_thickness)
    x = (frame.shape[1] - text_width) // 2  # Center text horizontally
    y = frame.shape[0] - text_height - 20  # Position near the bottom with some padding

    cv2.putText(
        frame,
        label_text, 
        (x, y),
        label_font, 
        label_scale, 
        (255, 255, 255), 
        label_thickness,
    )

    for _ in range(pause_frames):
        # Write the labeled frame to the output video
        out.write(frame)
    
    f_count += 1

video.release()
out.release()

print("Video predicted as:", results)