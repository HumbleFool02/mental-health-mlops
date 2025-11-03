from setuptools import setup, find_packages

setup(
    name="mental_health_mlops",
    version="0.1.0",
    description="MLOps pipeline for mental health sentiment analysis",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.1.4",
        "numpy>=1.26.3",
        "scikit-learn>=1.3.2",
        "mlflow>=2.9.2",
        "dvc>=3.37.0",
        "fastapi>=0.109.0",
        "transformers>=4.36.2",
        "torch>=2.1.2",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)