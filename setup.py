import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="roma",
    version="1.2.1",
    author="Romain Brégier",
    author_email="romain.bregier@naverlabs.com",
    description="A lightweight library to deal with 3D rotations in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/naver/roma",
    packages=["roma"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)