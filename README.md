# Weight Predictor ✨

A beautiful **Streamlit** web application that predicts weight based on height using a pre-trained **Linear Regression** model.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)

## Features

- 🎯 **ML-Powered Prediction** — Uses a trained Linear Regression model to predict weight from height
- 📏 **Flexible Input** — Enter height in Feet & Inches, Centimeters, or Inches (auto-converts to inches)
- ⚖️ **Dual Units** — Displays predicted weight in both **lbs** and **kg**
- 📊 **Interactive Chart** — Plotly spline curve showing weight vs height with your data point highlighted
- 📋 **Reference Table** — Quick-lookup table with your row highlighted
- 🏥 **BMI Calculator** — Auto-computes BMI with color-coded category badge
- 🎨 **Pinterest-Inspired UI** — Glassmorphism cards, gradient accents, micro-animations

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Astryn25/Weight-Predictor-Streamlit.git
   cd Weight-Predictor-Streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core language |
| Streamlit | Web framework |
| scikit-learn | ML model (Linear Regression) |
| Plotly | Interactive charts |
| joblib | Model serialization |

## Project Structure

```
├── app.py                          # Main Streamlit application
├── Best_model_Weight_Height.pkl    # Pre-trained Linear Regression model
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## How It Works

The app loads a pre-trained `LinearRegression` model from scikit-learn. The model takes height (in inches) as input and predicts weight (in pounds). The app handles unit conversion so you can enter height in your preferred unit.

## License

MIT License
