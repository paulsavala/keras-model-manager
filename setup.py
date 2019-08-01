import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras-model-manager",
    version="0.0.1",
    author="Paul Savala",
    author_email="paulsavala@gmail.com",
    description="A Python package to manage Keras Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paulsavala/keras-model-manager",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)