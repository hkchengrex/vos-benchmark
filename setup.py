from setuptools import setup

setup(
    name='gmp_benchmark',
    version='0.1.0',
    author='Rex Cheng',
    author_email='hkchengrex@gmail.com',
    packages=['gmp_benchmark'],
    url='http://pypi.python.org/pypi/PackageName/',
    license='LICENSE',
    description='General multi-threaded benchmarking script for video object segmentation.',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'Pillow',
        'tqdm',
        'opencv-python',
        'scikit-image',
    ],
)