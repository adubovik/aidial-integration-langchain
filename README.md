# Overview

`langchain_openai` [doesn't allow](https://github.com/langchain-ai/langchain/issues/26617) to pass extra request/response parameters to/from the upstream model.

The repo provides ways to overcome this issue.

## Minimal example

Find the minimal example highlighting the issue with `langchain_openai` at the [example folder](./example/):

```sh
> python -m venv .venv
> source ./.venv/bin/activate
> pip install -r requirements.txt
> python -m app
(1) Missing per-message request extra
(3) Missing per-message response extra
(4) Missing top-level response extra
```

`langchain_openai` ignores certain extra fields, meaning that the upstream endpoint won't receive (1) and the client won't receive (3) and (4) if they were sent by the upstream.

Note that **top-level request extra fields** do actually reach the endpoint.

## Solution #1 *(monkey-patching the library)*

One way to *fix* the issue, is to modify the methods which ignore these extra fields and make the methods actually take them into account.

This is achieved via monkey-patching certain private methods in `langchain_openai` which do the conversion from the Langchain datatypes to dictionaries and vice versa.

### Usage

Copy [the patch modules](./aidial_integration_langchain/patch/) to your project, then import before any Langchain module is imported:

```python
import patch # isort:skip  # noqa: F401

# ./example/app.py code
```

### Supported versions

The following `langchain_openai` versions have been tested for Python 3.9, 3.10, 3.11 and 3.12:

|Version|Request per-message|Response per-message|Response top-level|
|---|---|---|---|
|0.1.1|🟢|🟢|🔴|
|0.1.2|🟢|🟢|🔴|
|0.1.3|🟢|🟢|🔴|
|0.1.4|🟢|🟢|🔴|
|0.1.5|🟢|🟢|🔴|
|0.1.6|🟢|🟢|🔴|
|0.1.7|🟢|🟢|🔴|
|0.1.8|🟢|🟢|🔴|
|0.1.9|🟢|🟢|🔴|
|0.1.10|🟢|🟢|🔴|
|0.1.11|🟢|🟢|🔴|
|0.1.12|🟢|🟢|🔴|
|0.1.13|🟢|🟢|🔴|
|0.1.14|🟢|🟢|🔴|
|0.1.15|🟢|🟢|🔴|
|0.1.16|🟢|🟢|🔴|
|0.1.17|🟢|🟢|🔴|
|0.1.19|🟢|🟢|🔴|
|0.1.20|🟢|🟢|🔴|
|0.1.22|🟢|🟢|🔴|
|0.1.23|🟢|🟢|🟢|
|0.1.24|🟢|🟢|🟢|
|0.1.25|🟢|🟢|🟢|
|0.2.0|🟢|🟢|🟢|

Note that `langchain_openai<=0.1.22` doesn't support response top-level extra fields, since the structure of the code back then was not very amicable for monkey-patching in this particular respect.

## Solution #2 *(custom AzureChatOpenAI class)*

The implementation of the `AzureChatOpenAI` class may be copied and modified as needed to take into account extra fields.

Find the redefined classes at [aidial_integration_langchain.langchain_openai](./aidial_integration_langchain/langchain_openai/).

### Usage

Simply import the `AzureChatOpenAI` class from this repo instead of `langchain_openai`:

```diff
# ./example/app.py
- from langchain_openai import AzureChatOpenAI
+ from aidial_integration_langchain.langchain_openai import AzureChatOpenAI
```

### Supported versions

Currently only `langchain_openai==0.2.0` is supported for Python 3.9, 3.10, 3.11 and 3.12.