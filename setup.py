from setuptools import setup, find_packages

setup(
    name='classifier',
    version='',
    packages=find_packages(),
    url='',
    license='',
    author='Time IA-FRONT',
    author_email='',
    description='',
    install_requires=[
        "tensorflow==2.1.0",
        "click==7.1.2",
        "tensorflow-hub==0.9.0"
    ],
    entry_points={
        'console_scripts': [
            'train-classifier=classifier.training.main:main'
        ]
    }
)
