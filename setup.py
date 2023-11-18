from setuptools import find_packages, setup

setup(
    name="models",
    version="1.0.0",
    author="Johanna Sommer, Leon Hetzel",
    url="",
    packages=["models/"] + find_packages(),
    zip_safe=False,
    include_package_data=True,
)
