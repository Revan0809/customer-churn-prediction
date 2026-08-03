// Vanilla JS frontend for the Customer Churn Predictor.
// Talks to the local FastAPI backend -- no frameworks, no API keys.

// Local dev talks to the local backend; anywhere else (e.g. the deployed
// static site on Render) talks to the deployed backend. Update the
// production URL below once you know your actual Render backend URL.
const API_BASE_URL =
  location.hostname === "localhost" || location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "https://churn-backend-zrz7.onrender.com";

const form = document.getElementById("churn-form");
const submitBtn = document.getElementById("submit-btn");
const errorBox = document.getElementById("error-box");
const resultBox = document.getElementById("result");
const predictionTile = document.getElementById("prediction-tile");
const predictionValue = document.getElementById("prediction-value");
const probabilityValue = document.getElementById("probability-value");
const factorsList = document.getElementById("factors-list");
const modelInfoBox = document.getElementById("model-info-box");

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function buildPayload(formData) {
  return {
    gender: formData.get("gender"),
    SeniorCitizen: Number(formData.get("SeniorCitizen")),
    Partner: formData.get("Partner"),
    Dependents: formData.get("Dependents"),
    tenure: Number(formData.get("tenure")),
    PhoneService: formData.get("PhoneService"),
    MultipleLines: formData.get("MultipleLines"),
    InternetService: formData.get("InternetService"),
    OnlineSecurity: formData.get("OnlineSecurity"),
    OnlineBackup: formData.get("OnlineBackup"),
    DeviceProtection: formData.get("DeviceProtection"),
    TechSupport: formData.get("TechSupport"),
    StreamingTV: formData.get("StreamingTV"),
    StreamingMovies: formData.get("StreamingMovies"),
    Contract: formData.get("Contract"),
    PaperlessBilling: formData.get("PaperlessBilling"),
    PaymentMethod: formData.get("PaymentMethod"),
    MonthlyCharges: Number(formData.get("MonthlyCharges")),
    TotalCharges: Number(formData.get("TotalCharges")),
  };
}

function renderPrediction(data) {
  const isChurn = data.churn_prediction === "Yes";

  predictionTile.classList.remove("churn-yes", "churn-no");
  predictionTile.classList.add(isChurn ? "churn-yes" : "churn-no");
  predictionValue.textContent = isChurn ? "Yes" : "No";
  probabilityValue.textContent = `${(data.churn_probability * 100).toFixed(1)}%`;

  factorsList.innerHTML = "";
  if (data.top_factors.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No factor breakdown available for this prediction.";
    factorsList.appendChild(li);
  } else {
    data.top_factors.forEach((factor) => {
      const li = document.createElement("li");

      const label = document.createElement("span");
      label.textContent = factor.feature;

      const direction = document.createElement("span");
      const increases = factor.direction.includes("increases");
      direction.className = `factor-direction ${increases ? "increases" : "decreases"}`;
      direction.textContent = factor.direction;

      li.appendChild(label);
      li.appendChild(direction);
      factorsList.appendChild(li);
    });
  }

  resultBox.classList.remove("hidden");
}

async function handleSubmit(event) {
  event.preventDefault();
  clearError();
  resultBox.classList.add("hidden");

  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";

  try {
    const formData = new FormData(form);
    const payload = buildPayload(formData);

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const detail =
        typeof errorData.detail === "string"
          ? errorData.detail
          : JSON.stringify(errorData.detail) || `Request failed with status ${response.status}`;
      throw new Error(detail);
    }

    const data = await response.json();
    renderPrediction(data);
  } catch (err) {
    showError(
      `Could not get a prediction: ${err.message}. Is the backend running at ${API_BASE_URL}?`
    );
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict Churn";
  }
}

async function loadModelInfo() {
  try {
    const response = await fetch(`${API_BASE_URL}/model-info`);
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    const data = await response.json();

    const table = document.createElement("table");
    table.innerHTML = `
      <thead>
        <tr>
          <th>Model</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
          <th>ROC-AUC</th>
        </tr>
      </thead>
      <tbody></tbody>
    `;
    const tbody = table.querySelector("tbody");

    data.comparison.forEach((row) => {
      const tr = document.createElement("tr");
      if (row.model === data.best_model_name) {
        tr.classList.add("best-model");
      }
      tr.innerHTML = `
        <td>${row.model}${row.model === data.best_model_name ? " (selected)" : ""}</td>
        <td>${row.precision.toFixed(3)}</td>
        <td>${row.recall.toFixed(3)}</td>
        <td>${row.f1.toFixed(3)}</td>
        <td>${row.roc_auc.toFixed(3)}</td>
      `;
      tbody.appendChild(tr);
    });

    modelInfoBox.innerHTML = "";
    modelInfoBox.appendChild(table);
  } catch (err) {
    modelInfoBox.innerHTML = `<p class="error-box">Could not load model info. Is the backend running at ${API_BASE_URL}?</p>`;
  }
}

form.addEventListener("submit", handleSubmit);
loadModelInfo();
