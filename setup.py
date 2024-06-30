from setuptools import setup

setup(name='scribblescreator',
    version='1.3.0',
    description='Automatic scribble creation based on ground truth data.',
    author='Roman Schwob',
    author_email='roman.schwob@students.unibe.ch',
    license='GNU GPLv3',
    packages=['.', 'scribbles_helpers'],
    package_dir={'': 'src'},
    zip_safe=False)