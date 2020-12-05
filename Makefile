help:
	@echo "Commands:"
	@echo ""
	@echo "  flake8       style check with flake8"
	@echo "  doc          build docs (update existing)"
	@echo "  doc-clean    build docs (new, removing any existing)"
	@echo "  preview      renders docs in Browser"
	@echo "  linkcheck    check all links in docs"
	@echo "  deploy       deploy gallery to gh-pages"
	@echo "  clean        clean up all generated files"
	@echo ""

flake8:
	flake8 docs/conf.py examples/

doc:
	cd docs && make html && cd ..

doc-clean:
	cd docs && rm -rf gallery/ && rm -rf _build/ && make html && cd ..

preview:
	xdg-open docs/_build/html/index.html &

linkcheck:
	cd docs && make html -b linkcheck && cd ..

deploy:
	bash deploy.sh

clean:
	rm -rf docs/gallery/ docs/_build/
	rm -rf examples/comparisons/raw.githubusercontent.com/
	rm -rf examples/interactions/GemPy-II-topo.npy
	rm -rf examples/time_domain/*.npz
	rm -rf examples/tutorials/*.h5 examples/tutorials/*.cfg
	rm -rf examples/tutorials/*.log examples/tutorials/*.json
