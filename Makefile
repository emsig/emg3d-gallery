help:
	@echo "Commands:"
	@echo ""
	@echo "  install       install conda-env emg3d-gallery"
	@echo "  flake8        style check with flake8"
	@echo "  doc           build docs (update existing)"
	@echo "  example FILE= build particular example"
	@echo "  doc-clean     build docs (new, removing any existing)"
	@echo "  preview       renders docs in Browser"
	@echo "  linkcheck     check all links in docs"
	@echo "  deploy        deploy gallery to gh-pages (as is; run doc before)"
	@echo "  clean         clean up all generated files"
	@echo "  remove        remove conda-env emg3d-gallery"
	@echo ""

.ONESHELL:
install:
	conda env create -f environment.yml

remove:
	conda remove --name emg3d-gallery --all

flake8:
	flake8 docs/conf.py examples/

doc:
	cd docs && make html

doc-clean:
	cd docs && rm -rf gallery/*/ && rm -rf _build/ && make html

example:
	cd docs && sphinx-build -D sphinx_gallery_conf.filename_pattern=$(FILE) -b html -d _build/doctrees . _build/html

preview:
	xdg-open docs/_build/html/index.html

linkcheck:
	cd docs && make html -b linkcheck

.ONESHELL:
deploy:
	mkdir tmp
	cp -r docs/_build/html/* tmp/.
	cp -r .git tmp/.
	cd tmp/
	touch .nojekyll
	git branch -D gh-pages &>/dev/null
	git checkout --orphan gh-pages
	git add --all
	git commit -m 'Update gallery'
	git push -f --set-upstream origin gh-pages
	cd ..
	rm -rf tmp/

clean:
	rm -rf docs/gallery/*/ docs/gallery/*.zip docs/_build/
	rm -rf examples/comparisons/raw.githubusercontent.com/
	rm -rf examples/comparisons/*.h5
	rm -rf examples/models/GemPy-II-topo.npy
	rm -rf examples/t-domain/*.npz
	rm -rf examples/tutorials/*.h5 examples/tutorials/*.cfg
	rm -rf examples/tutorials/*.log examples/tutorials/*.json
