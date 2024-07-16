from setuptools import setup

setup(name='scribblescreator',
    version='2.0.1',
    description='Automatic scribble creation based on ground truth data.',
    author='Roman Schwob',
    author_email='roman.schwob@students.unibe.ch',
    license='GNU GPLv3',
    packages=['.', 'scribbles_testing'],
    package_dir={'': 'src'},
    zip_safe=False)