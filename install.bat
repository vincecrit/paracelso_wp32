@ECHO OFF

ECHO CREO AMBIENTE WP32
python -m venv wp32

CALL wp32\Scripts\activate.bat

ECHO INSTALLO DIPENDENZE
pip install numpy scipy pandas h5py pyyaml lxml bs4 opencv-python rasterio geopandas Pillow scikit-image

pip freeze > requirements.txt