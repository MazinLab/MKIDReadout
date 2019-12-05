from __future__ import print_function

import os
import platform
import subprocess

import numpy
import setuptools
import sys
from Cython.Build import cythonize
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.extension import Extension


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
    
    #don't compile if installing on anything that's not linux
    if platform.system()!='Linux':
        return

    src_paths = ['./mkidreadout/readout/mkidshm']
                #'./mkidreadout/readout/packetmaster/']

    # compile the software
    cmds = ["gcc -shared -o libmkidshm.so -fPIC mkidshm.c -lrt -lpthread"]
    venv = get_virtualenv_path()

    try:
        for cmd, src_path in zip(cmds, src_paths):
            if venv:
                cmd += ' --prefix=' + os.path.abspath(venv)
            subprocess.check_call(cmd, cwd=src_path, shell=True)
    except Exception as e:
        print(str(e))
        raise e


class CustomInstall(install, object):
    """Custom handler for the 'install' command."""
    def run(self):
        compile_and_install_software()
        super(CustomInstall,self).run()


class CustomDevelop(develop, object):
    """Custom handler for the 'install' command."""
    def run(self):
        compile_and_install_software()
        super(CustomDevelop,self).run()


extensions = [Extension(name="mkidreadout.channelizer.roach2utils",
                        sources=['mkidreadout/channelizer/roach2utils.pyx'],
                        include_dirs=[numpy.get_include()],
                        extra_compile_args=['-fopenmp'],
                        extra_link_args=['-fopenmp']),
              Extension(name="mkidreadout.readout.sharedmem",
                        sources=['mkidreadout/readout/sharedmem.pyx'],
                        include_dirs=[numpy.get_include(), 'mkidreadout/readout/mkidshm'],
                        extra_compile_args=['-shared', '-fPIC'],
                        library_dirs=['mkidreadout/readout/mkidshm'],
                        runtime_library_dirs=[os.path.abspath('mkidreadout/readout/mkidshm')],
                        extra_link_args=['-O3', '-lmkidshm', '-lrt', '-lpthread']),# '-Wl,-rpath=mkidreadout/readout/mkidshm']),
              Extension(name="mkidreadout.readout.packetmaster",
                        sources=['mkidreadout/readout/pmthreads.c', 'mkidreadout/readout/packetmaster.pyx'],
                        include_dirs=[numpy.get_include(), 'mkidreadout/readout/packetmaster',
                                      'mkidreadout/readout/mkidshm'],
                        library_dirs=['mkidreadout/readout/mkidshm'],
                        runtime_library_dirs=[os.path.abspath('mkidreadout/readout/mkidshm')],
                        extra_compile_args=['-O3', '-shared', '-fPIC'],
                        extra_link_args=['-lmkidshm', '-lrt', '-lpthread'])
             ]

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="mkidreadout",
    version="0.7.1",
    author="MazinLab",
    author_email="mazinlab@ucsb.edu",
    description="An UVOIR MKID Data Readout Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MazinLab/MKIDReadout",
    packages=setuptools.find_packages(),
    package_data={'mkidreadout': ('config/*.yml', 'resources/firmware/*', 'resources/firfilters/*')},
    scripts=['mkidreadout/channelizer/initgui.py',
             'mkidreadout/channelizer/hightemplar.py',
             'mkidreadout/readout/dashboard.py',
             'mkidreadout/channelizer/reinitADCDAC.py',
             'mkidreadout/configuration/powersweep/clickthrough_hell.py',
             'mkidreadout/configuration/beammap/sweep.py'],
    ext_modules=cythonize(extensions),
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research"
    ),
    cmdclass={'install': CustomInstall,'develop': CustomDevelop}
)



