from setuptools import find_packages, setup

package_name = 'event_reader'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jonurce',
    maintainer_email='jonurce@todo.todo',
    description='Read events from rosbag',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'print_events = event_reader.event_read:main',
        ],
    },
)
