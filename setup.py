"""Package setup"""

from setuptools import setup, find_packages


setup(
    name="hoo",
    version="0.1.0",
    description="Algorithms that derive from Hierarchical Optimistic Optimization (HOO)",
    long_description_content_type="text/markdown",
    setup_requires=["setuptools>=38.6.0"],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gym==0.26.2",
        "matplotlib==3.7.1",
        "pydantic==1.10.4",
    ],
)
