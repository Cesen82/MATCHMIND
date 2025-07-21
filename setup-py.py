"""
ProFootballAI Package Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="profootball-ai",
    version="2.0.0",
    author="ProFootballAI Team",
    author_email="support@profootball-ai.com",
    description="Professional AI-powered football predictions for Over 2.5 goals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/profootball-ai",
    project_urls={
        "Documentation": "https://docs.profootball-ai.com",
        "Issues": "https://github.com/yourusername/profootball-ai/issues",
        "Source": "https://github.com/yourusername/profootball-ai",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "ml": [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "lightgbm>=4.0.0",
        ],
        "visualization": [
            "dash>=2.14.0",
            "dash-bootstrap-components>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "profootball-ai=main:main",
            "profootball-train=src.models.train:main",
            "profootball-api=src.api.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "profootball_ai": [
            "data/*.json",
            "data/*.csv",
            "models/*.pkl",
            "ui/assets/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "football",
        "soccer",
        "predictions",
        "machine-learning",
        "sports-analytics",
        "betting",
        "over-2.5-goals",
        "ai",
        "streamlit",
    ],
)