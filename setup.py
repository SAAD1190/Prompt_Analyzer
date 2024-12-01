from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the contents of the requirements file
with open('requirements.txt', 'r', encoding='utf-8') as f:
    required = f.read().splitlines()

setup(
    name='Prompt_Analyzer',
    version='0.1.0',
    author='GIIA_DS',
    author_email='',
    description='An advanced tool for analyzing and processing prompts with semantic, syntactic, and readability metrics.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SAAD1190/Prompt_Analyzer',
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        'Development Status :: 3 - Alpha',  # Indicate the development stage
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    python_requires='GNU GENERAL PUBLIC LICENSE', 
    keywords='prompt analysis, semantic analysis, readability, syntax', 
    project_urls={ 
        'Bug Tracker': '', 
        'Documentation': '',
        'Source Code': '',
    },
)
