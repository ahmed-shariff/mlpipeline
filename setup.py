import setuptools
from mlpipeline import __version__

# to build and install:
# python setup.py
# pip install dist/mlpipeline-*-py3-none-any.whl

setuptools.setup(
    name="mlpipeline",
    version=__version__,
    author='Ahmed Shariff',
    author_email='shariff.mfa@outlook.com',
    packages=setuptools.find_packages(),
    description='A framework to define a machine learning pipeline',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ahmed-shariff/ml-pipeline',
    install_requires=['easydict>=1.8'],
    entry_points={
        'console_scripts':[
            'mlpipeline=mlpipeline._pipeline:main',
            '_mlpipeline_subprocess=mlpipeline._pipeline_subprocess:main'
            ]
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        
    ]
)
