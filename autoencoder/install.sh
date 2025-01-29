#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install required modules
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    pip install tensorflow[and-cuda]
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" || "$OSTYPE" == "mingw32" || "$OSTYPE" == "mingw64" ]]; then
    pip install tensorflow-gpu
else
    echo "Could not find supported OS"
fi

pip install pillow
pip install tensorboard-plugin-customizable-plots
pip install matplotlib
pip install seaborn
pip install tqdm

sudo apt install jq