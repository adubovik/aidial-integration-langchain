import nox

nox.options.reuse_existing_virtualenvs = True

SRC = "."


def format_with_args(session: nox.Session, *args):
    session.run("autoflake", *args)
    session.run("isort", *args)
    session.run("black", *args)


@nox.session
def lint(session: nox.Session):
    """Runs linters and fixers"""
    try:
        session.run("poetry", "install", "--all-extras", external=True)
        session.run("poetry", "check", "--lock", external=True)
        session.run("pyright", SRC)
        session.run("flake8", SRC)
        format_with_args(session, SRC, "--check")
    except Exception:
        session.error(
            "linting has failed. Run 'make format' to fix formatting and fix other errors manually"
        )


@nox.session
def format(session: nox.Session):
    """Runs linters and fixers"""
    session.run("poetry", "install", external=True)
    format_with_args(session, SRC)


supported_python_versions = [
    "3.9",
    "3.10",
    "3.11",
    "3.12",
]
supported_lc_openai_versions = [
    # "0.1.22",  # fail: cannot import name '_convert_chunk_to_generation_chunk' from 'langchain_openai.chat_models.base'
    "0.1.23",
    "0.1.24",
    "0.1.25",
    "0.2.0",
]


@nox.session(python=supported_python_versions)
@nox.parametrize("langchain_openai", supported_lc_openai_versions)
def test_monkey_patch(session: nox.Session, langchain_openai: str) -> None:
    """Runs tests for the patch"""
    session.run("poetry", "install", external=True)
    session.install(f"langchain_openai=={langchain_openai}")
    session.run(
        "pytest",
        "tests/test_langchain_monkey_patch.py",
        "tests/test_langchain_noop.py",
    )


@nox.session(python=supported_python_versions)
@nox.parametrize("langchain_openai", ["0.2.0"])
def test_custom_class(session: nox.Session, langchain_openai: str) -> None:
    """Runs tests for the patch"""
    session.run("poetry", "install", external=True)
    session.install(f"langchain_openai=={langchain_openai}")
    session.run("pytest", "tests/test_langchain_custom_class.py")


@nox.session(python=supported_python_versions)
@nox.parametrize("openai", ["1.48.0"])
def test_openai(session: nox.Session, openai: str) -> None:
    """Runs tests for the patch"""
    session.run("poetry", "install", external=True)
    session.install(f"openai=={openai}")
    session.run("pytest", "tests/test_openai.py")
