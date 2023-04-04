python -m venv venv
.\venv\Scripts\activate.ps1
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn run:app --reload --port 8123