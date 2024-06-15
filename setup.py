#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name='pvat',
    version='0.0.0.dev2',
    description=(
        'Python implementations of variance reduction techniques for'
        ' extensive-form games'
    ),
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    url='https://github.com/uoftcprg/pvat',
    author='University of Toronto Computer Poker Student Research Group',
    author_email='uoftcprg@studentorg.utoronto.ca',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Education',
        'Topic :: Games/Entertainment',
        'Topic :: Games/Entertainment :: Board Games',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    keywords=[
        'aivat',
        'artificial-intelligence',
        'deep-learning',
        'divat',
        'game',
        'game-development',
        'game-theory',
        'holdem-poker',
        'imperfect-information-game',
        'libratus',
        'mivat',
        'pluribus',
        'poker',
        'python',
        'reinforcement-learning',
        'texas-holdem',
    ],
    project_urls={
        'Documentation': 'https://pvat.readthedocs.io/en/latest/',
        'Source': 'https://github.com/uoftcprg/pvat',
        'Tracker': 'https://github.com/uoftcprg/pvat/issues',
    },
    packages=find_packages(),
    install_requires='numpy>=1.26.4,<2',
    python_requires='>=3.11',
    package_data={'pvat': ['py.typed']},
)
