from setuptools import find_packages, setup

setup(
    name="mental_health_mlops",
    version="0.1.0",
    description="MLOps pipeline for mental health sentiment analysis",
    author="Piyush Kasle",
    author_email="kaslepiyush07@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.1.4",
        "numpy>=2.0.0,<3.0.0",
        "scikit-learn>=1.3.2",
        "fastapi>=0.109.0",
        "transformers>=4.36.2",
        "torch>=2.1.2",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.13",
    ],
)
