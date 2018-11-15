import setuptools
from mlpipeline import __version__

setuptools.setup(
    name="mlpipeline",
    version=__version__,
    author='Ahmed Shariff',
    author_email='shariff.mfa@outlook.com',
    packages=setuptools.find_packages(),
    description='A framework to define a machine learning pipeline',
    long_description=open('README.md').read(),
    url='https://github.com/ahmed-shariff/ml-pipeline',
    entry_points={
        'console_scripts':[
            'mlpipeline=mlpipeline.pipeline:main',
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
