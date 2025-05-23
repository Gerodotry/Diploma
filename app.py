from flask import Flask, render_template, request, url_for
from ultralytics import YOLO, SAM
from moviepy import VideoFileClip
from transformers import pipeline
import torch, numpy as np, cv2, os, csv

app = Flask(__name__)

yolo_model = YOLO('/runs/detect/train/weights/best.pt')
sam_model = SAM("sam2_b.pt")

asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model="Liubun/wav2vec2-xls-r-300m-uk",
    return_timestamps="word"
)

def load_bad_words(file_path):
    bad_words = set()
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for word in row:
                cleaned = word.strip().lower()
                if cleaned:
                    bad_words.add(cleaned)
    return bad_words

BAD_WORDS = load_bad_words("synonyms.csv")

@app.route('/', methods=['GET', 'POST'])
def index():
    result_video = None
    transcript = []
    bad_words_info = []

    if request.method == 'POST':
        file = request.files['video']
        if file:
            video_path = os.path.join("uploads", file.filename)
            file.save(video_path)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = os.path.join("static", "processed.mp4")
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            frame_idx = 0
            fall_class_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % 10 == 0:
                    results = yolo_model(frame)
                    for result in results:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        for (x1, y1, x2, y2), cls in zip(boxes, classes):
                            if int(cls) == fall_class_id:
                                sam_result = sam_model.predict(frame, bboxes=[[x1, y1, x2, y2]])
                                mask = sam_result[0].masks.data[0].cpu().numpy()
                                mask_uint8 = (mask * 255).astype('uint8')
                                colored_mask = np.zeros_like(frame)
                                colored_mask[:, :, 2] = mask_uint8
                                frame = cv2.addWeighted(frame, 1, colored_mask, 0.3, 0)
                out.write(frame)
                frame_idx += 1
            cap.release()
            out.release()
            result_video = url_for('static', filename='processed.mp4')

            video_clip = VideoFileClip(video_path)
            audio_path = os.path.join("uploads", "audio.wav")
            video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
            video_clip.close()

            result = asr_pipeline(audio_path)
            for entry in result["chunks"]:
                word = entry["text"]
                start = round(entry["timestamp"][0], 2)
                end = round(entry["timestamp"][1], 2)
                transcript.append(f"{word} [{start}-{end}]")
                if word.lower() in BAD_WORDS:
                    bad_words_info.append({
                        "word": word,
                        "start": start,
                        "end": end
                    })

    return render_template('index.html',
                           video_url=result_video,
                           transcript=transcript,
                           bad_words=bad_words_info)

if __name__ == '__main__':
    app.run(debug=True)
