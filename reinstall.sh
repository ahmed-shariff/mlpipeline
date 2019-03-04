# Simple helper script to reinstall and test the package
python setup.py sdist bdist_wheel
pip uninstall mlpipeline -y
pip install dist/mlpipeline-1.1a3.post8-py3-none-any.whl
