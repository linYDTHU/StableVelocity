from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stablevs",
    version="0.1.0",
    author="Donglin Yang",
    author_email="ydlin718@gmail.com",
    description="StableVS: Stable Velocity Sampling Extensions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/linYDTHU/StableVelocity/StableVS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=[
        "diffusers>=0.36.0",
        "torch>=2.0.0",
        "numpy",
        "scipy",
        "Pillow",
        "accelerate",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
)
