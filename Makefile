VENV_DIR ?= .venv
UV ?= $(VENV_DIR)/bin/uv
UV_VERSION ?= 0.7.3

all: install

init_env:
	python -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install uv==$(UV_VERSION) --quiet

install: init_env
	$(UV) sync --quiet --all-extras

build: install
	$(UV) build

publish: build
	$(UV) publish --no-sources

clean:
	rm -rf $(VENV_DIR)
	rm -rf .nox
	rm -rf .pytest_cache
	rm -rf dist
	find . -type d -name __pycache__ | xargs rm -r

lint: install
	$(UV) run -- nox -s lint

format: install
	$(UV) run -- nox -s format

test: install
	$(UV) run -- nox -s test_monkey_patch $(if $(PYTHON),--python=$(PYTHON),)
	$(UV) run -- nox -s test_custom_class $(if $(PYTHON),--python=$(PYTHON),)
	$(UV) run -- nox -s test_openai $(if $(PYTHON),--python=$(PYTHON),)

help:
	@echo '===================='
	@echo 'build                        - build the library'
	@echo 'clean                        - clean virtual env and build artifacts'
	@echo 'publish                      - publish the library to Pypi'
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo '-- TESTS --'
	@echo 'test                         - run unit tests'
	@echo 'test PYTHON=<python_version> - run unit tests with the specific python version'
