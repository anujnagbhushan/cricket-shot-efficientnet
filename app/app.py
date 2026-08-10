from flask import Flask, render_template, request, jsonify
from inference.predict_video import predict as predict_video
from app.logger import log, get_logs
from app.database import initialize_database, get_connection
import os
import subprocess

app = Flask(__name__)

# Create database automatically on startup
initialize_database()

UPLOAD_FOLDER = "app/static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("app/static/ai_vision", exist_ok=True)


@app.route("/")
def index():
    return render_template("index_v3.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded."})

    video = request.files["video"]

    filepath = os.path.join(
        UPLOAD_FOLDER,
        video.filename
    )

    log("Upload request received")
    video.save(filepath)
        # ----------------------------------------
    # Convert uploaded video to MP4
    # ----------------------------------------

    converted_path = os.path.join(
        "app",
        "static",
        "uploads",
        "original.mp4"
    )

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i",
        filepath,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        converted_path
    ], check=True)

    log("Video saved successfully")

    log("Starting AI pipeline")

    result = predict_video(converted_path)
    result["original_video"] = "/static/uploads/original.mp4"
    log("Analysis completed successfully", "success")

    return jsonify(result)


@app.route("/logs")
def logs():
    return jsonify(get_logs())


@app.route("/save_result", methods=["POST"])
def save_result():

    data = request.json

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analysis (
            session_id,
            batsman_name,
            match_format,
            playing_hand,
            prediction,
            confidence,
            inference_time,
            coach_correct,
            correct_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["session_id"],
        data["batsman_name"],
        data["match_format"],
        data["playing_hand"],
        data["prediction"],
        data["confidence"],
        data["inference_time"],
        data["coach_correct"],
        data["correct_label"]
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "status": "saved",
        "message": "Result saved successfully!"
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)