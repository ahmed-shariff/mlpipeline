rm -rf build
rm -rf dist
python setup.py sdist
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
