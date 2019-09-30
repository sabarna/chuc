from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='chuc',
    version='0.2',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Conversion Homogeneity based Uplift computation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy','pandas', 'sklearn', 'xgboost', 'pylift', 'seaborn','matplotlib', 'scipy'],
    url='https://github.com/sabarna/chuc',
    author='Sabarna Choudhuri',
    author_email='sabarna121@gmail.com'
)
