# Ensure all examples were run.
cd docs
make html
cd ..

# Create tmp-dir
tmp_dir=$(mktemp -d -t emg3d-gallery-XXXXXXXXXX)
echo $tmp_dir
cpwd=$(pwd)
gallery=$(basename "$PWD")
cp -rf $cpwd/ $tmp_dir/.
cd $tmp_dir/$gallery

# Move html to root directory
mv docs/_build/html .

# Stash current status
git stash

# Delete existing gh-pages branch and create new one
git branch -D gh-pages
git checkout --orphan gh-pages

# Remove all files but 'html' and 'git'
rm -rf docs/ examples/ README.rst LICENSE .gitignore
rm -rf environment.yml Makefile deploy.sh

# Extract webpage, delete html dir.
mv html/* .
rm -r html

# Create nojekyll
touch .nojekyll

# Upload site
git add --all
git commit -m 'Update gallery'
git push -f --set-upstream origin gh-pages

# Clean up
rm -rf $tmp_dir
