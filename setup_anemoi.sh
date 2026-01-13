#!/bin/bash
export PYTHON_HOME=/leonardo_work/DestE_330_25/users/asalihi0/compiled-libraries/python/python-3.11.7-gcc-12.2.0-cmake-3.27.9
export SQLITE3_HOME=/leonardo_work/DestE_330_25/users/asalihi0/compiled-libraries/python/sqlite-3.45-gcc-12.2.0
export PATH=$PYTHON_HOME/bin:$SQLITE3_HOME/bin:$PATH
export LD_LIBRARY_PATH=$PYTHON_HOME/lib:$SQLITE3_HOME/lib:$LD_LIBRARY_PATH
python3 -m venv multi-domain-torch-2.6.0-cu124
source multi-domain-torch-2.6.0-cu124/bin/activate
python3 -m pip install "numpy<=2"
python3 -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python3 -m pip install trimesh
python3 -m pip install -e ./graphs
python3 -m pip install -e ./models
python3 -m pip install -e ./training
python3 -m pip install -e bris-inference