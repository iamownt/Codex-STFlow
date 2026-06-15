from setuptools import find_namespace_packages, setup

setup(
    name='stflow',
    packages=find_namespace_packages(include=['stflow', 'stflow.*']),
    package_data={
        'stflow.hest_utils': [
            'local_ckpts.json',
            'pretrained_configs/*.json',
        ],
    },
)
