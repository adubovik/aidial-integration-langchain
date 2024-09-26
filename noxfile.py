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
    # "3.9",
    # "3.10",
    "3.11",
    # "3.12",
]
supported_lc_openai_versions = [
    # "0.0.2",  # AsyncCompletions.create() got an unexpected keyword argument 'http_async_client'
    "0.1.1",
    "0.1.2",
    "0.1.3",
    "0.1.4",  # module 'langchain_openai.chat_models.base' has no attribute 'BaseChatOpenAI'
    "0.1.5",
    "0.1.6",
    "0.1.7",
    "0.1.8",
    "0.1.9",
    "0.1.10",
    "0.1.11",
    "0.1.12",
    "0.1.13",
    "0.1.14",
    "0.1.15",
    "0.1.16",  # BaseChatOpenAI._create_chat_result() takes 2 positional arguments but 3
    "0.1.17",
    "0.1.19",
    "0.1.20",
    "0.1.22",  # '_convert_chunk_to_generation_chunk' from 'langchain_openai.chat_models.base' doesn't exist
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
