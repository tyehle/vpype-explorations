from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="vpype-explorations",
    version="0.1.0",
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Antoine Beyeler",
    url="https://github.com/abey79/vpype-explorations/",
    license=license,
    packages=find_packages(exclude=("examples", "tests")),
    install_requires=[
        "axi @ git+https://github.com/fogleman/axi",
        "click",
        "vpype @ git+https://github.com/abey79/vpype.git",
        "shapely",
        "numpy",
        "scipy",
        "scikit-image",
        "opencv-python",
    ],
    entry_points="""
            [vpype.plugins]
            alien=vpype_explorations.alien:alien
            fracture=vpype_explorations.fracture:fracture
            variablewidth=vpype_explorations.variablewidth:variablewidth
            mdgrid=vpype_explorations.mdgrid:mdgrid
            msimage=vpype_explorations.moduleset:msimage
            msrandom=vpype_explorations.moduleset:msrandom
            msfingerprint=vpype_explorations.moduleset:msfingerprint
            mstiles=vpype_explorations.moduleset:mstiles
            fake3d=vpype_explorations.fake3d:fake3d
            spiro=vpype_explorations.spiro:spiro
            poly=vpype_explorations.poly:poly
        """,
)
