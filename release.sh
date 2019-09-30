#!/usr/bin/env bash
python -m setup.py sdist bdist_wheel
python -m twine upload dist/*

