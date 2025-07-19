Local opensource google font finder from image.
## WIP

1) Database Creation + Vectorizing Fonts ` notebooks/db_creatuon.ipynb `
2) Find similar fonts ` notebooks/recognition_pipeline.ipynb `

### Start
python3.12:
` pip install -r requirements.txt
python3 notebooks/create_database.py`

### For training
python3.12:
` pip install -r requirements.txt
python3 notebooks/create_database.py
python3 notebooks/font_dataset.py`