from setuptools import setup, find_packages

setup(
    name="flight_price_predictor",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pydantic>=2.0.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "advanced": [
            "torch>=2.0.0",
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
        ],
    },
)
