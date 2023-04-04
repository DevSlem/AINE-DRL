from setuptools import find_packages, setup

setup(
    name="aine-drl",
    version="0.1.0",
    description="AINE-DRL is a deep reinforcement learning (DRL) baseline framework. AINE means \"Agent IN Environment\".",
    author="DevSlem",
    author_email="devslem12@gmail.com",
    packages=find_packages(include=["aine_drl"]),
    install_requires=[
        "torch==1.11.0",
        "tensorboard==2.12.0",
        "PyYAML==6.0",
        "gym==0.26.2",
        "gym[all]",
        "mlagents==0.30.0",
        "protobuf==3.20.3",
    ],
    python_requires=">=3.9",
)
