from setuptools import setup, find_packages

setup(
    name="stock-prophet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "pandas-ta>=0.3.14b0",
        "python-telegram-bot",
        "statsmodels",
        "telegram",
        "twilio",
        "yfinance",
    ],
    python_requires=">=3.9",
    description="A sophisticated stock analysis and forecasting application",
    author="Mr. Otmane",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)