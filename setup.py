# https://packaging.python.org/tutorials/distributing-packages/

import evgena
from setuptools import setup, find_packages

setup(
    name='evgena',
    version=evgena.__version__,
    description='Evolutionary Generated Adversarial Examples',
    long_description='42',  # TODO enhance
    url='https://github.com/proste/evgena',
    author='Stepan Prochazka',
    author_email='prochazka.stepan@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords='42',  # TODO enhance
    packages=find_packages(),
    install_requires=[
        'deap>=1.0.0',
        'scikit-learn>=0.19.0',
        'keras>=2.0.0'
        # TODO add if needed
    ],
    python_requires='>=3'
    # TODO consider adding packae_data or data_files
)