console.log("Cricket Shot AI Dashboard Loaded");

const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("video");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const time = document.getElementById("time");

const statusText = document.getElementById("statusText");

const originalVideo = document.getElementById("originalVideo");
const aiVideo = document.getElementById("aiVideo");

const topPredictions =
    document.getElementById("topPredictions");

const summary =
    document.getElementById("summary");

const terminal =
    document.getElementById("terminal");



form.addEventListener("submit", async (e)=>{

    function addTerminalLog(time, message, level = "info") {


    const line = document.createElement("div");

    line.className = "terminal-line";

    line.innerHTML = `
        <span class="time">${time}</span>
        <span class="${level}">${message}</span>
    `;

    terminal.appendChild(line);

    terminal.scrollTop = terminal.scrollHeight;
}


async function pollLogs() {

    try {

        const response = await fetch("/logs");

        const logs = await response.json();

        logs.forEach(log => {

            addTerminalLog(
                log.time,
                log.message,
                log.level
            );

        });

    } catch (err) {

        console.error(err);

    }

}

    e.preventDefault();

    if(fileInput.files.length===0){

        alert("Please choose a cricket video.");

        return;

    }

    //----------------------------------------
    // Loading UI
    //----------------------------------------

    statusText.innerText="Uploading Video...";
    terminal.innerHTML = "";

addTerminalLog("--:--:--", "Starting analysis...");

const logInterval = setInterval(pollLogs, 200);

    prediction.innerText="--";
    confidence.innerText="--";
    time.innerText="--";



    //----------------------------------------
    // Show Original Video
    //----------------------------------------

    //----------------------------------------
// Show Original Video
//----------------------------------------

const file = fileInput.files[0];

originalVideo.pause();
originalVideo.removeAttribute("src");
originalVideo.load();
//----------------------------------------
// Send Video
//----------------------------------------

const formData = new FormData();

formData.append(
    "video",
    fileInput.files[0]
);

const response = await fetch("/predict", {

    method: "POST",

    body: formData

});




    const data=
        await response.json();

        clearInterval(logInterval);

await pollLogs();

    console.log(data);

    //----------------------------------------
// Original Video (converted MP4)
//----------------------------------------

originalVideo.src = data.original_video + "?" + Date.now();

originalVideo.load();

originalVideo.onloadedmetadata = () => {
    originalVideo.currentTime = 0;
    originalVideo.play().catch(console.error);
};
        //----------------------------------------
    // Update Statistics
    //----------------------------------------

    prediction.innerText =
        data.prediction;

    confidence.innerText =
        data.confidence.toFixed(2) + "%";

    time.innerText =
    Number(data.inference_time).toFixed(2) + " sec";



    //----------------------------------------
    // AI Vision
    //----------------------------------------

    aiVideo.src =
        "/static/ai_vision/AI_Vision.mp4?" + Date.now();

    aiVideo.load();



    //----------------------------------------
    // Status
    //----------------------------------------

    statusText.innerText =
        "COMPLETED";



    //----------------------------------------
    // Match Summary
    //----------------------------------------

    // Match Summary removed in v3 UI
if (summary) {
summary.innerHTML = `

    <div class="summary-row">

        <span>Shot</span>

        <strong>${data.prediction}</strong>

    </div>

    <div class="summary-row">

        <span>Confidence</span>

        <strong>${data.confidence.toFixed(2)}%</strong>

    </div>

    <div class="summary-row">

        <span>Inference</span>

        <strong>${data.inference_time} sec</strong>

    </div>

    <div class="summary-row">

        <span>Frames</span>

        <strong>30</strong>

    </div>

    <div class="summary-row">

        <span>Model</span>

        <strong>EfficientNet-B3 + GRU</strong>

    </div>

    <div class="summary-row">

        <span>Status</span>

        <strong>Completed</strong>

    </div>

`;
}




    //----------------------------------------
    // Top Predictions
    //----------------------------------------

    topPredictions.innerHTML = "";

data.top5.forEach((item, index) => {

    const row = document.createElement("tr");

    row.innerHTML = `
        <td>${index + 1}</td>
        <td>${item.shot}</td>
        <td>${item.confidence.toFixed(2)}%</td>
    `;

    topPredictions.appendChild(row);

});
});

// ==============================
// SAVE RESULT
// ==============================

const saveBtn = document.getElementById("saveBtn");

saveBtn.addEventListener("click", async () => {

    if (prediction.innerText === "--") {
        alert("Please analyze a video first.");
        return;
    }

    const payload = {

        session_id: document.getElementById("sessionId").value,

        batsman_name: document.getElementById("batsmanName").value,

        match_format: document.getElementById("matchFormat").value,

        playing_hand: document.getElementById("battingHand").value,

        prediction: prediction.innerText,

        confidence: parseFloat(
            confidence.innerText.replace("%", "")
        ),

        inference_time: parseFloat(
            time.innerText.replace(" sec", "")
        ),

        coach_correct: document.getElementById("predictionCorrect").value,

        correct_label: document.getElementById("correctLabel").value

    };

    try {

        const response = await fetch("/save_result", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)

        });

        const data = await response.json();

        alert(data.message);

    } catch (err) {

        console.error(err);

        alert("Failed to save result.");

    }

});

