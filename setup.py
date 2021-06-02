from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name="jax_dataclasses",
    version="1.0.0",
    description="Dataclasses + JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/brentyi/jax_dataclasses",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="MIT",
    packages=find_packages(),
    package_data={"jax_dataclasses": ["py.typed"]},
    python_requires=">=3.7",
    install_requires=[
        "flax",
        "jax",
        "jaxlib",
    ],
    extras_require={
        "testing": [
            "pytest",
            "pytest-cov",
            # "hypothesis",
            # "hypothesis[numpy]",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
