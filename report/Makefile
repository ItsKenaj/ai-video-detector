# Makefile for CS229 Final Report

MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz

cleanall: clean
	rm -f $(MAIN).pdf

.PHONY: all clean cleanall

