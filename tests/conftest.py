import pytest

from verifaith import Verifier
from verifaith.claims import SentenceExtractor
from verifaith.testing import KeywordNLI


@pytest.fixture
def nli():
    return KeywordNLI()


@pytest.fixture
def verifier(nli):
    return Verifier(SentenceExtractor(), nli)
