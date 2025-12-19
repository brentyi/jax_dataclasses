from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name="jax_dataclasses",
    version="1.6.3",
    description="Dataclasses + JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/brentyi/jax_dataclasses",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="MIT",
    packages=find_packages(),
    package_data={"jax_dataclasses": ["py.typed"]},
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.25",  # For `jax.tree_util.default_registry`
        "jaxlib",
        "typing_extensions",
    ],
    extras_require={
        "testing": [
            "flax",  # Used for serialization tests.
            "pytest",
            "pytest-cov",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
