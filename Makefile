ADIOS_DIR = /Users/guanyisun/Documents/adios-1.13.1/bin
LIB_DIR = lib
TEST_DIR = test
DIR = $(shell pwd)
TEST_SCRIPT_NAME = run_tests.sh

default: build-test-script

build-test-script:
	echo '#!/bin/bash' > $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'platform=`uname`' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'if [[ $$platform == "Darwin" ]]; then ' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo '    script_dir=$(DIR)/$(TEST_DIR)' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'else' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo '    script_dir=$$(dirname "$$(readlink -f "$$0")")' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'fi' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'export PATH=$(ADIOS_DIR):$$PATH' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'export PYTHONPATH=$(DIR)/$(LIB_DIR):$$PATH:$$PYTHONPATH' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'cd $(DIR)/$(TEST_DIR)' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'python3 parser_test.py' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	echo 'python3 milof_test.py' >> $(TEST_DIR)/$(TEST_SCRIPT_NAME)
	chmod +x $(TEST_DIR)/$(TEST_SCRIPT_NAME)

test: build-test-script
	bash $(TEST_DIR)/$(TEST_SCRIPT_NAME)
