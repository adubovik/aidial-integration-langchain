PY_FACTOR=${PYTHON:+py${PYTHON//./}}
echo "PYTHON=$PYTHON"
echo "PY_FACTOR=$PY_FACTOR"
"$UV" run -- tox -p 4 --parallel-no-spinner -f test_openai $PY_FACTOR
"$UV" run -- tox -p 4 --parallel-no-spinner -f test_custom_class $PY_FACTOR
"$UV" run -- tox -p 4 --parallel-no-spinner -f test_monkey_patch $PY_FACTOR