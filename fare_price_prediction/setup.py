
from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List:

    requirements = []

    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="fare price prediction",
    version="0.0.1",
    description="this project is created to predict the fare prices of famous CAB service company.",
    author="Surakshit",
    author_email="surakshitkapoor.developer@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
)