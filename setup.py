from setuptools import setup, find_packages


def get_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "A Scikit-Learn Compatible Python library for ANFIS."


setup(
    name="scikit-anfis",
    version="1.2.1",
    description = 'A Scikit-Learn Compatible Python library for ANFIS',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy >= 1.12.0",
        "scipy >= 1.0.0",
        "Jinja2 >= 3.1.2",
        "threadpoolctl >= 3.2.0",
        "torch >= 2.1.0",
        "typing_extensions >= 4.8.0",
        "tzdata >= 2023.3",
        "pandas >= 2.1.1",
        "matplotlib >= 3.8.0",
        "mpmath >= 1.3.0",
        "networkx >= 3.1",
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    python_requires=">=3.8",
)