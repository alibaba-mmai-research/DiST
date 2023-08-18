cd ..
rm -rf code.tar.gz;
tar -zc --exclude='*.git' --exclude='.vscode' --exclude='output' --exclude='current_epoch' --exclude='core-python-*' --exclude='core-0-python-*' --exclude='*.pyth' --exclude='visualization' --exclude='__pycache__' -f code.tar.gz HiCo requirements.txt