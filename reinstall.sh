# Simple helper script to reinstall and test the package
python setup.py sdist bdist_wheel
pip uninstall mlpipeline -y
find dist -name "mlpipeline-*.whl" -exec pip install {} \;

