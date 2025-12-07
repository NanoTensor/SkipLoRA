from setuptools import setup, find_packages

setup(
    name="skiplora",
    version="0.1.0",
    description="SkipLoRA: Contextual Gradient Zeroing for Accelerated LoRA Fine-Tuning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "peft>=0.6.0",  # For base LoRA integration
        "datasets>=2.14.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)