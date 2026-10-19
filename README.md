# 🚦 TrafficPulse

> A Machine Learning web application that predicts road traffic congestion levels — built as a B.Tech 2nd-year CSE mini-project.

---

## 📌 What Does This Project Do?

TrafficPulse lets you enter basic road conditions (time of day, day of week, road name, number of vehicles, average speed) and instantly predicts whether traffic is **Low**, **Medium**, or **High**.

It uses a **Logistic Regression** model trained on simulated traffic data. The prediction happens through a simple website — no app installs needed. You fill in a form, click Predict, and the result appears on screen along with a probability bar chart.

---

## 🛠️ Technologies Used

| Technology | What It Does |
|---|---|
| **Python 3.10+** | Core programming language for the ML pipeline and web server |
| **Flask** | Lightweight Python web framework — serves the webpage and handles predictions |
| **scikit-learn** | Provides the Logistic Regression model, StandardScaler, and evaluation tools |
| **NumPy** | Handles numerical arrays; scikit-learn models require NumPy arrays as input |
| **joblib** | Saves and loads the trained model (`model.pkl`) and scaler (`scaler.pkl`) |
| **HTML / CSS / JavaScript** | Frontend — the webpage the user sees and interacts with |
| **CSV (Python built-in)** | Stores and reads the training dataset without needing pandas |

---

## 📁 Project Folder Structure

```
TrafficPulse/
│
├── app.py                  # Flask web server — handles routes and predictions
├── generate_dataset.py     # Generates simulated traffic data → traffic_data.csv
├── train_model.py          # Trains the ML model → saves model.pkl and scaler.pkl
├── feature_config.py       # Shared road and day constants for encoding
├── requirements.txt        # Python packages needed to run the project
├── Procfile                # Tells cloud hosting (Render) how to run the web server
├── render.yaml             # Render deployment configuration
├── .gitignore              # Files to exclude from Git
│
├── traffic_data.csv        # Generated training dataset
├── model.pkl               # Saved trained model
├── scaler.pkl              # Saved feature scaler
│
├── templates/
│   ├── index.html          # Main single-page prediction dashboard
│   └── about.html          # Dedicated documentation & architecture page
│
└── static/
    ├── style.css           # Custom transportation-inspired styling
    └── script.js           # AJAX fetch(), Chart.js analytics, dynamic probability bars
```

---

## ⚙️ Setup: How to Get the Project Running

### Step 1 — Install Python 3.10 or higher

Download from [https://www.python.org/downloads/](https://www.python.org/downloads/)

During installation, check the box **"Add Python to PATH"**.

Verify installation by opening a terminal and running:
```bash
python --version
```

### Step 2 — Open the Project in VS Code

1. Open VS Code.
2. Go to **File → Open Folder** and select the `TrafficPulse` folder.
3. Open the **integrated terminal**: `View → Terminal` (or press `` Ctrl+` ``).

### Step 3 — Install Required Packages

In the VS Code terminal, run:

```bash
python -m pip install -r requirements.txt
```

This installs Flask, NumPy, scikit-learn, and joblib. Wait for it to finish.

---

## 📊 Step 4 — Generate the Dataset

```bash
python generate_dataset.py
```

**What this does:**
- Creates a file called `traffic_data.csv` in the project folder.
- The file contains thousands of simulated traffic records (hour, day, road name, vehicle count, speed, and a congestion label).
- This is **simulated data** — not real traffic — but it follows realistic rules (e.g., rush hours are busier).

---

## 🤖 Step 5 — Train the Model

```bash
python train_model.py
```

**What this does:**
- Reads `traffic_data.csv`.
- Preprocesses the data (one-hot encodes road names and days, scales all numbers).
- Trains a Logistic Regression model using 80% of the data.
- Tests the model on the remaining 20% and prints the accuracy.
- Saves the trained model as `model.pkl` and the scaler as `scaler.pkl`.

**What the accuracy output means:**

```
Model Accuracy: 0.87
```

This means the model correctly predicted the traffic level for **87% of the test samples**. For this project, anything above 80% is considered good.

---

## 🚀 Step 6 — Run the Web Application

```bash
python app.py
```

You should see output like:

```
 * Running on http://127.0.0.1:5000
 * Press CTRL+C to quit
```

Open your browser and go to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

> ⚠️ Keep the terminal open while using the app. Closing it stops the server.

---

## 🖥️ How to Use the Website

1. **Fill in the form:**
   - **Hour of Day** — Enter a number from 0 (midnight) to 23 (11 PM)
   - **Day of Week** — Select from the dropdown (Monday to Sunday)
   - **Road Name** — Choose one of the 5 available roads
   - **Vehicles per Hour** — Approximate number of vehicles (e.g., 300)
   - **Average Speed (km/h)** — Typical speed on that road right now

2. **Click "Predict Traffic".**

3. **View the result:**
   - A coloured badge shows the prediction: 🟢 Low / 🟡 Medium / 🔴 High
   - A **bar chart** shows the probability for each class (how confident the model is)
   - A short **recommendation** is displayed (e.g., "Consider an alternate route")

4. **Try different combinations** to see how hour and vehicle count affect the prediction.

---

## 🐛 Common Errors and How to Fix Them

### ❌ `ModuleNotFoundError: No module named 'flask'`
**Fix:** You haven't installed the dependencies yet. Run:
```bash
python -m pip install -r requirements.txt
```

---

### ❌ `FileNotFoundError: model.pkl not found`
**Fix:** You need to train the model first. Run these in order:
```bash
python generate_dataset.py
python train_model.py
python app.py
```

---

### ❌ `Address already in use` / `Port 5000 is in use`
**Fix:** Another program (or a previous Flask instance) is using port 5000.
- On **Windows**: Open Task Manager → find `python.exe` → End Task, then try again.
- Or change the port in `app.py`:
```python
app.run(debug=True, port=5001)
```
Then open `http://127.0.0.1:5001` instead.

---

### ❌ `500 Internal Server Error` in the browser
**Fix:** Check the terminal where Flask is running — it will show the full Python error message. Common causes:
- `model.pkl` or `scaler.pkl` is missing (re-run `train_model.py`)
- A field was left empty or sent an unexpected value
- A code bug — read the traceback carefully in the terminal

---

## ⚠️ Limitations

- **Simulated data only:** The dataset is artificially generated, not collected from real roads. The model learns the rules we coded, not real-world complexity.
- **No live traffic feed:** The app predicts based on user input — it doesn't pull real-time data from any source.
- **Limited roads and scenarios:** Only 5 roads are supported. Real traffic systems handle thousands.
- **Model simplicity:** Logistic Regression is a baseline model. More complex models (Random Forest, XGBoost) might perform better on real-world data.
- **No persistence:** Predictions are not stored anywhere. Refreshing the page clears everything.

---

## 📢 Disclaimer

This project is built **for educational purposes only** as a B.Tech 2nd-year CSE mini-project. The traffic predictions are based on **simulated, synthetic data** and should **not** be used for real navigation, urban planning, or any decision-making purposes. Results do not reflect actual road conditions.

---

*Made with ❤️ for learning — TrafficPulse © 2024*
