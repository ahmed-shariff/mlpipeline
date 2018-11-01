import setuptools


setuptools.setup(
    name="mlpipeline",
    version='1.1.a.3',
    author='Ahmed Shariff',
    author_email='shariff.mfa@outlook.com',
    packages=setuptools.find_packages(),
    description='A framework to define a machine learning pipeline',
    long_description=open('README.md').read(),
    url='https://github.com/ahmed-shariff/ml-pipeline',
    entry_points='''
        [console_scripts]
        mlpipeline=mlpipeline.pipeline:main
    ''',
    classifiers=[
        "Programming Language :: Python :: 3"]
)



