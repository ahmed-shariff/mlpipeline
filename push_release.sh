rm -rf build
rm -rf dist
python setup.py sdist bdist
twine upload dist/*
