#!/bin/bash

module load GCCcore/11.3.0
module load Python/3.10.4

cd semseg
echo $PWD

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
