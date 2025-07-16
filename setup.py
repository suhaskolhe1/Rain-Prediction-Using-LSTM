from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->list[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements=[req.replace("\n", "") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements


setup(
    name="Rain Prediction Using LSTM",
    version="0.0.1",
    author="Suhas Kolhe",
    author_email="suhaskolhe1111@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)