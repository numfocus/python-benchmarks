WEB_REPO_ALIAS  ?= origin

all: clean run github

clean:
	rm -f report/*.json
	rm -f report/*.html

run:
	python run_benchmarks.py

github:
	@echo "Send to github"
	ghp-import -p report -r ${WEB_REPO_ALIAS}
