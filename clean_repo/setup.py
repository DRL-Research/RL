"""Editable install for the hierarchical multi-agent coordination library.

    pip install -e .

exposes the two importable packages used throughout the scripts:
  * ``src``        — models, training loop, experiment config, diagnostics, plotting
  * ``highwayenv`` — custom multi-agent highway-env environments (REL* gym ids)
"""
from setuptools import setup, find_packages

setup(
    name="hierarchical-marl-coordination",
    version="1.0.0",
    description="3-level hierarchical multi-agent RL for coordinated driving in highway-env",
    packages=find_packages(include=["src", "src.*", "highwayenv", "highwayenv.*", "logger", "logger.*"]),
    python_requires=">=3.10",
    install_requires=[
        "stable-baselines3==2.8.0a2",
        "gymnasium==1.2.3",
        "highway-env==1.10.2",
        "torch>=2.4",
        "numpy>=2.0",
        "scipy",
        "pandas",
        "matplotlib",
        "pygame",
    ],
)
