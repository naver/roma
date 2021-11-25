#!/bin/bash

# Build documentation
bash build_doc.sh

cd docs/
# Empty repo
git init
#Create GitHub pages branch
git checkout -b gh-pages

# Add docs file
git add -A
git commit -m "Documentation"

# Add origin and push
git remote add origin git@github.com:naver/roma.git
git push --force origin gh-pages

cd ..