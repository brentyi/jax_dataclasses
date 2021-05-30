from setuptools import find_packages, setup

setup(
    name="jax_dataclasses",
    version="0.0",
    description="Dataclasses + JAX",
    url="http://github.com/brentyi/jax_dataclasses",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="BSD",
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
            # "pytest-cov",
            # "hypothesis",
            # "hypothesis[numpy]",
        ]
    },
)
