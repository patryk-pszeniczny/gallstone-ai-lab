# Gallstone AI Lab 🧪

Python project for detecting gallstones (cholelithiasis) from clinical-style data using two approaches: a classic MLP model and a fuzzy-logic engine. On top of that there’s a small GUI so you don’t have to run everything from the terminal. This is a **study/demo** repo — good for uni, presentations, showing code structure. Not medical software.

---

## Features

-  **MLP-based classification** (scikit-learn style)
-  **Fuzzy logic** alternative with CSV export
-  **GUI layer** to enter patient data and get a prediction
-  **Excel input** (`dataset-uci.xlsx`)
-  Clean project layout (`core/` = logic, `gui/` = UI)

---

## Project structure

```text
gallstone-ai-lab/
├── main.py                         # entrypoint, wires everything together
├── core/                           # ML / fuzzy / data logic
├── gui/                            # UI layer
├── assets/                         # icons, images for GUI
├── dataset-uci.xlsx                # sample/input dataset
└── gallstone_fuzzy_predictions.csv # sample output from fuzzy engine
```

## Installation
```text
git clone https://github.com/patryk-pszeniczny/gallstone-ai-lab.git
cd gallstone-ai-lab

python -m venv venv
# Windows:
# venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

