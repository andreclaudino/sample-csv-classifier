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
        "tensorflow>=2.6.0",
        "click==7.1.2",
        "tensorflow-hub==0.9.0",
        "tensorflow_text>=2.0.0rc0"
    ],
    entry_points={
        'console_scripts': [
            'train-classifier=classifier.training.main:main'
        ]
    }
)
