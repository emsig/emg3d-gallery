# Clean-up previous runs and run all.
cd docs
make clean
rm -rf gallery
make html
cd ..

# Move html to root directory
mv docs/_build/html .

# Stash current status
git stash

# Delete existing gh-pages branch and create new one
git branch -D gh-pages
git checkout --orphan gh-pages

# Remove all files but 'html' and 'git'
rm -rf docs/ examples/ README.rst LICENSE .gitignore deploy.sh

# Extract webpage, delete html dir.
mv html/* .
rm -r html

# Create nojekyll and htaccess
touch .nojekyll

# Upload site
git add --all
git commit -m 'Update gallery'
git push -f --set-upstream origin gh-pages

# Leave it on master again
git checkout master
