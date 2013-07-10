WEB_REPO_ALIAS  ?= origin

all: clean run

clean:
	rm -f report/*.json
	rm -f report/*.html

run:
	python run_benchmarks.py

github:
	@echo "Publish report to github.io pages"
	ghp-import -p report -r ${WEB_REPO_ALIAS}
