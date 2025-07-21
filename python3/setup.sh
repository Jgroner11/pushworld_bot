python3.10 -m venv venv
source venv/bin/activate
pip install --require-hashes -r requirements.txt
echo $(python3 -c 'import os; print(os.getcwd())')/src > $(python3 -c 'import site; print(site.getsitepackages()[0])')/pushworld.pth
