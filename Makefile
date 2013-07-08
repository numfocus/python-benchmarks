WEB_REPO_ALIAS  ?= origin

all: run github

run:
	python run_benchmarks.py

github:
	@echo "Send to github"
	ghp-import -p report -r ${WEB_REPO_ALIAS}
