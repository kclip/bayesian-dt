from setuptools import setup, find_packages

setup(
    name="dt-mpr-channel",
    packages=find_packages(),
    entry_points='''
        [console_scripts]
        cli=cli:cli
    ''',
)