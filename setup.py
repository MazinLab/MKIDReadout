import setuptools, sys
import os
from setuptools.command.install import install
import subprocess
#pip install -e git+http://github.com/mazinlab/mkidreadout.git@restructure#egg=mkidreadout --src ./mkidtest


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
    src_path = './mkidreadout/mkidreadout/readout/packetmaster/'

    # compile the software
    cmds = ["gcc -Wall -Wextra -o packetmaster packetmaster.c -I. -lm -lrt -lpthread -O3"]
#            'gcc -o Bin2PNG Bin2PNG.c -I. -lm -lrt -lpng',
#            'gcc -o BinToImg BinToImg.c -I. -lm -lrt',
#            'gcc -o BinCheck BinCheck.c -I. -lm -lrt']
    venv = get_virtualenv_path()

    print os.getcwd()
    try:

        for cmd in cmds:
            if venv:
                cmd += ' --prefix=' + os.path.abspath(venv)
            subprocess.check_call(cmd, cwd=src_path, shell=True)
    except Exception as e:
        print str(e)
        raise e

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
    scripts=['mkidreadout/channelizer/InitGui.py',
             'mkidreadout/channelizer/HighTemplar.py',
             'mkidreadout/readout/MkidDashboard.py'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research"
    ),
    cmdclass={'install': CustomInstall}
)



