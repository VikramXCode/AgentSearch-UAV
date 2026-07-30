alert("script loaded");

// =============================
// DOM Elements
// =============================

const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const resultImage = document.getElementById("resultImage");

const detectBtn = document.getElementById("detectBtn");
const queryInput = document.getElementById("query");

const target = document.getElementById("target");
const objects = document.getElementById("objects");
const confidence = document.getElementById("confidence");
const status = document.getElementById("status");

const tableBody = document.querySelector("#resultTable tbody");

// =============================
// Pipeline
// =============================

const pipeline = [
    "queryAgent",
    "knowledgeAgent",
    "strategyAgent",
    "srAgent",
    "sahiAgent",
    "detectAgent",
    "verifyAgent",
    "explainAgent"
];

// =============================
// Image Preview
// =============================

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) return;

    const reader = new FileReader();

    reader.onload = function (e) {

        previewImage.src = e.target.result;

        // Reset result image
        resultImage.src = e.target.result;

    };

    reader.readAsDataURL(file);

});

// =============================
// Reset Pipeline
// =============================

function resetPipeline() {

    pipeline.forEach(id => {

        const item = document.getElementById(id);

        item.classList.remove("active");
        item.classList.remove("waiting");
        item.classList.remove("failed");

    });

}

// =============================
// Animate Pipeline
// =============================

function animatePipeline() {

    resetPipeline();

    let i = 0;

    const interval = setInterval(() => {

        if (i > 0) {
            document
                .getElementById(pipeline[i - 1])
                .classList.remove("waiting");

            document
                .getElementById(pipeline[i - 1])
                .classList.add("active");
        }

        if (i < pipeline.length) {

            document
                .getElementById(pipeline[i])
                .classList.add("waiting");

            i++;

        } else {

            clearInterval(interval);

            finishDetection();

        }

    }, 800);

}

// =============================
// Fake Detection (Demo)
// Replace with Backend API Later
// =============================

async function finishDetection() {

    const file = imageInput.files[0];

    const formData = new FormData();
    formData.append("query", queryInput.value);
    formData.append("image", file);

    try {

        status.textContent = "Detecting...";

        const response = await fetch("http://127.0.0.1:5000/detect", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Server Error: " + response.status);
        }

        const data = await response.json();

        console.log(data);

        // Update Mission Summary
        target.textContent = data.target;
        objects.textContent = data.objects_found;
        confidence.textContent = data.confidence + "%";
        status.textContent = data.status;

        // Update Result Table
        tableBody.innerHTML = "";

        data.detections.forEach(det => {

            tableBody.innerHTML += `
                <tr>
                    <td>${det.object}</td>
                    <td>${det.confidence}%</td>
                </tr>
            `;

        });

        // Show image
        resultImage.src = previewImage.src;

    }
    catch (error) {

        console.error(error);

        status.textContent = "Error";

        alert(error.message);

    }

}
// =============================
// Detect Button
// =============================

detectBtn.addEventListener("click", async function (event) {

    event.preventDefault();

    console.log("Button clicked");

    await finishDetection();

    console.log("finishDetection completed");

});

// =============================
// Future Backend Integration
// =============================

/*

Replace finishDetection() with:

const formData = new FormData();

formData.append("query", queryInput.value);
formData.append("image", imageInput.files[0]);

fetch("http://127.0.0.1:5000/detect", {
    method: "POST",
    body: formData
})
.then(res => res.json())
.then(data => {

    target.textContent = data.target;
    objects.textContent = data.objects_found;
    confidence.textContent = data.confidence;
    status.textContent = data.status;

    resultImage.src = data.result_image;

    tableBody.innerHTML = "";

    data.detections.forEach(det => {

        tableBody.innerHTML += `
        <tr>
            <td>${det.object}</td>
            <td>${det.confidence}%</td>
        </tr>
        `;

    });

});

*/