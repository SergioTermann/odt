from setuptools import setup, find_packages

setup(
    name="gym_dogfight",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'harfang',
    ],
    author="bafs",
    description="A gym environment for dogfight simulation",
)