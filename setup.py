import setuptools

with open('README.md') as f:
    README = f.read()

with open("requirements.txt") as f:
    REQUIREMENTS = [line.strip() for line in f if line.strip()]

# with open("LICENSE") as f:
#     LICENSE = [line.strip() for line in f if line.strip()]

SOURCE_URL = 'https://github.com/qcware/qcware-unitair.git'

setuptools.setup(
    name="qcware-unitair",
    version="0.1.0",
    author="QC Ware Corp. (Sean Weinberg, Fabio Sanches)",
    author_email="sean.weinberg@qcware.com",
    description="PyTorch-based quantum computing.",
    long_description=README,
    long_description_content_type="text/markdown",
    url=SOURCE_URL,
    # license=LICENSE,
    packages=setuptools.find_packages(exclude=("tests",)),
    test_suite="tests",
    install_requires=REQUIREMENTS,
    include_package_data=True,
    package_data={
        "": ["*.typed"]
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7'
)
