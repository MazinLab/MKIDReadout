import setuptools, sys
import os
from setuptools.command.install import install
import subprocess


def get_virtualenv_path():
    """Used to work out path to install compiled binaries to."""
    if hasattr(sys, 'real_prefix'):
        return sys.prefix

    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        return sys.prefix

    if 'conda' in sys.prefix:
        return sys.prefix

    return None


def compile_and_install_software():
    """Used the subprocess module to compile/install the C software."""
    src_path = './mkidreadout/readout/packetmaster/'

    # compile the software
    cmd = "gcc -Wall -Wextra -o packetmaster packetmaster.c -I. -lm -lrt -lpthread -O3"
    venv = get_virtualenv_path()
    if venv:
        cmd += ' --prefix=' + os.path.abspath(venv)
    subprocess.check_call(cmd, cwd=src_path, shell=True)

    # install the software (into the virtualenv bin dir if present)
    subprocess.check_call('make install', cwd=src_path, shell=True)


class CustomInstall(install):
    """Custom handler for the 'install' command."""
    def run(self):
        compile_and_install_software()
        super().run()


with open("README.md", "r") as fh:
    long_description = fh.read()



setuptools.setup(
    name="mkidreadout",
    version="0.0.1",
    author="MazinLab",
    author_email="mazinlab@ucsb.edu",
    description="An UVOIR MKID Data Readout Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MazinLab/MKIDReadout",
    packages=setuptools.find_packages(),
    scripts=['mkidreadout/channelizer/InitGUI.py',
             'mkidreadout/channelizer/HighTemplar.py',
             'mkidreadout/channelizer/MkidDashboard.py']
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research"
    ),
    cmdclass={'install': CustomInstall}
)



