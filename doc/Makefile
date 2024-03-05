SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
NOTEBOOKSDIR  = ../example_notebooks/
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help Makefile

%: Makefile
	mkdir $(SOURCEDIR)/examples
	cp -r $(NOTEBOOKSDIR)/* $(SOURCEDIR)/examples/
	cp ../CHANGELOG $(SOURCEDIR)/changelog.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf $(SOURCEDIR)/examples

clean:
	rm -rf $(SOURCEDIR)/examples
	rm -rf $(SOURCEDIR)/autoapi
	rm -rf $(SOURCEDIR)/GaAs_crystal.xsf
	rm -rf $(SOURCEDIR)/test.h5
	rm -rf $(SOURCEDIR)/test_workflow.yaml
	rm -rf $(SOURCEDIR)/changelog.rst
	rm -rf $(BUILDDIR)
