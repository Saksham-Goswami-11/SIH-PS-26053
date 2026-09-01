import sys
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

sources = [
    "engine/src/grid_polar_cpu.cpp",
    "engine/src/hardware_detector.cpp",
    "engine/bindings/pybind_module.cpp",
]

include_dirs = ["engine/include"]
extra_compile_args = ["-O3", "-std=c++17"]

if sys.platform == "darwin":
    extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.15"]

ext_modules = [
    Pybind11Extension(
        "kavach_engine",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        cxx_std=17,
    ),
]

setup(
    name="kavach_engine",
    version="1.0.0",
    description="KAVACH-2.5D Foveated Elevation & Semantic Mapping Engine",
    packages=find_packages(include=["backend*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
