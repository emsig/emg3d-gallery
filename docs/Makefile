# Command line options.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = emg3d-gallery
SOURCEDIR     = .
BUILDDIR      = _build

# Will also be triggered if "make" is provided without argument.
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help html-noplot Makefile

html-noplot:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -D plot_gallery=0 $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# To run only a specific file:
# sphinx-build -D sphinx_gallery_conf.filename_pattern=filename.py -b html . _build/html
