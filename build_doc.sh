#!/bin/bash
cd docsource/
make html
cd ..
rm -rf docs/
cp -r docsource/build/html docs/
# Sources are actually not required for the documentation.
rm -r docs/_sources