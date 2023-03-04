# Writer Verification
Train a writer verification model using PyTorch and serve it in a Flask Webapp.

## Usage:

### Process Data
The model presented here is trained on [BanglaWriting](https://data.mendeley.com/datasets/r43wkvdk4w/1) and [BN-HTRd](https://data.mendeley.com/datasets/743k6dm543/1) datasets. First the two datasets are combined and annotated with writer id using the [data_processing.ipynb](data_processing.ipynb) script.

To process the data, simply run each cell of the notebook.

### Run Webserver
```
python webapp.py
```