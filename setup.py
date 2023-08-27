import setuptools

with open("README.md", "r",encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.1"

REPO_NAME = "boctor"
AUTHOR_NAME = "varadtechx"
AUTHOR_EMAIL = "varadrane1707@gmail.com"
SRC_REPO = "boctor"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description="Create your own chatbot using this library",
    long_description="Create your own chatbot using this library",
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": SRC_REPO},
    packages=setuptools.find_packages(where=SRC_REPO),
    install_requires=['nltk==3.8.1','numpy==1.22.0','torch==1.9.1'],
    )