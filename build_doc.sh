#!/bin/bash
cd docsource/
make html
cd ..
rm -rf docs/
cp -r docsource/build/html docs/