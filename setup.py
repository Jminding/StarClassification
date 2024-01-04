from setuptools import setup, find_packages
setup(
    name='MK-AI',
    version='0.1.3',
    author='Jaymin Ding',
    author_email='jtding43@gmail.com',
    description='MK-AI is a Python package that allows you to classify stars using the MK system and machine learning.',
    packages=find_packages(where="src"),
    package_dir = {"": "src"},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=['selenium', 'chromedriver_autoinstaller']
)