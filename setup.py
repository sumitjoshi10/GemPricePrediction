from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> List[str]:
    '''
    This function returns the list of requirements
    
    '''
    requirements = []
    with open(file_path,"r") as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

with open("README.md","r",encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"                       
PKG_NAME = "GEM_Price_Predictor"            # Package Name
AUTHER_USER_NAME="sumitjoshi10"             # Github Username
AUTHER_EMAIL = "sumit.joshi9818@gmail.com"  # Github Email
REPO_NAME = "GemPricePrediction"            # Github Repository name

setup(
    name=PKG_NAME,
    version=__version__,
    author=AUTHER_USER_NAME,
    author_email=AUTHER_EMAIL,
    description="Will Predict the Price of GEM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHER_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHER_USER_NAME}/{REPO_NAME}/issues"
    },
    # package_dir={"":"src"},
    # packages= find_packages(where="src"),
    packages= find_packages(),
    install_requires = get_requirements("requirements_dev.txt")
)