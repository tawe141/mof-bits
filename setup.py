from setuptools import setup


setup(
    name='mof-bits',
    version='0.1.0',
    description='A way to represent MOFs with bit vectors',
    url='https://github.com/tawe141/mof-bits',
    author='Eric Taw',
    author_email='tawe141@berkeley.edu',
    packages=['mofbits'],
    install_requires=[
        # 'rdkit',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Chemistry'
    ]
)
