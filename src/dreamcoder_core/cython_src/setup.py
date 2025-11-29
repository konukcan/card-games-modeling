#!/usr/bin/env python3
"""
Setup script for building Cython modules.

Usage:
    python setup.py build_ext --inplace

This will compile the .pyx files to .so files that can be imported.
"""

from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
import os

# Cython compiler options for maximum performance
Options.docstrings = False
Options.embed_pos_in_docstring = False

# Directory containing this setup.py
here = os.path.dirname(os.path.abspath(__file__))

# List of Cython modules to build
cython_modules = [
    "type_system_cy.pyx",
    "program_cy.pyx",
    "grammar_cy.pyx",
    "enumeration_cy.pyx",
    "lean_primitives_cy.pyx",  # Cython-native primitives
]

# Filter to only existing files
existing_modules = [m for m in cython_modules if os.path.exists(os.path.join(here, m))]

if not existing_modules:
    print("No .pyx files found to compile!")
    exit(1)

print(f"Building Cython modules: {existing_modules}")

setup(
    name="dreamcoder_cython",
    version="1.0.0",
    description="Cythonized DreamCoder core modules",
    ext_modules=cythonize(
        existing_modules,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
            'nonecheck': False,
        },
        annotate=True,  # Generate HTML annotation files
    ),
    zip_safe=False,
)
