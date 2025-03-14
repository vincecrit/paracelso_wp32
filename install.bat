@ECHO OFF

ECHO CREO AMBIENTE WP32
python -m venv wp32

CALL wp32\Scripts\activate.bat

ECHO INSTALLO DIPENDENZE
pip install -r requirements.txt