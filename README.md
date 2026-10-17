# Student Risk Prediction Dashboard

An interactive student risk prediction dashboard using a Random Forest classifier implemented in JavaScript in the browser.

## Features
- Overview charts and risk distribution
- Live student risk prediction and recommendations
- Random Forest training, metrics and feature importance
- Search/filter students, CSV import and export
- Locally saved added students

## Run locally
Open `index.html` in your browser. An internet connection is needed to load Chart.js from its CDN.

## Publish with GitHub Pages
1. Create a public GitHub repository.
2. Upload `index.html` and `README.md` to the repository root.
3. Go to **Settings → Pages** and select **Deploy from a branch**, `main`, `/ (root)`.
4. Open the published GitHub Pages URL after deployment.

## Notes
The starting student dataset is synthetic (300 generated students). This is a demonstration model, not a validated academic intervention tool. The Random Forest is implemented in client-side JavaScript; no Python backend is required.
