# Ensure all examples were run.
cd docs
make html
cd ..

# Copy relevant bits to temporary direction, move there.
mkdir tmp
cp -r docs/_build/html/* tmp/.
cp -r .git tmp/.
cd tmp

# Stash current status
git stash

# Delete existing gh-pages branch and create new one
git branch -D gh-pages
git checkout --orphan gh-pages

# Create nojekyll and htaccess
touch .nojekyll

# Upload site
git add --all
git commit -m 'Update gallery'
git push -f --set-upstream origin gh-pages

# Move back and remove temporary direction.
cd ..
rm -rf tmp
