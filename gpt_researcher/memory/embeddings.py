import os
import logging
from typing import Any

OPENAI_EMBEDDING_MODEL = os.environ.get(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

_SUPPORTED_PROVIDERS = {
    "openai",
    "azure_openai",
    "cohere",
    "gigachat",
    "google_vertexai",
    "google_genai",
    "fireworks",
    "ollama",
    "together",
    "mistralai",
    "huggingface",
    "nomic",
    "voyageai",
    "dashscope",
    "custom",
    "bedrock",
    "aimlapi",
    "netmind",
}


class Memory:
    def __init__(self, embedding_provider: str, model: str, **embedding_kwargs: Any):
        from gpt_researcher.utils.budget import current_research_budget, ResearchBudgetError
        budget = current_research_budget.get()
        if budget is not None:
            budget.raise_if_denied()
            custom_clients = any(embedding_kwargs.get(key) is not None for key in
                                 ("client", "async_client", "http_client", "http_async_client"))
            endpoint = (embedding_kwargs.get("openai_api_base") or embedding_kwargs.get("base_url")
                        or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")
            if (embedding_provider != "openai" or model != "text-embedding-3-small" or custom_clients
                    or endpoint not in {"https://api.openai.com/v1", "https://api.openai.com/v1/"}):
                if budget.mode == "enforce":
                    raise ResearchBudgetError("budget_invalid_transition")
                logging.getLogger(__name__).warning("Shadow research embedding provider is not metered")
            else:
                proxy = embedding_kwargs.get("openai_proxy", os.getenv("OPENAI_PROXY"))
                embedding_kwargs.update(budget.http_clients(proxy=proxy))
                embedding_kwargs["openai_proxy"] = ""
                # Keep length-safe tokenization/splitting, but bound vector
                # response size and per-request reservation, not research size.
                embedding_kwargs["chunk_size"] = 16
                if budget.mode == "enforce":
                    embedding_kwargs["max_retries"] = 0
        _embeddings = None
        match embedding_provider:
            case "custom":
                from langchain_openai import OpenAIEmbeddings
                from gpt_researcher.credentials import CredentialConfigError, vault_mode

                if vault_mode():
                    # It would read the env copy of the managed OpenAI key.
                    raise CredentialConfigError("The custom embedding provider is not available with Vault credentials")

                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("OPENAI_API_KEY", "custom"),
                    openai_api_base=os.getenv(
                        "OPENAI_BASE_URL", "http://localhost:1234/v1"
                    ),  # default for lmstudio
                    check_embedding_ctx_length=False,
                    **embedding_kwargs,
                )  # quick fix for lmstudio
            case "openai":
                from langchain_openai import OpenAIEmbeddings

                # Support custom OpenAI-compatible APIs via OPENAI_BASE_URL
                if "openai_api_base" not in embedding_kwargs and os.environ.get("OPENAI_BASE_URL"):
                    embedding_kwargs["openai_api_base"] = os.environ["OPENAI_BASE_URL"]

                from gpt_researcher.credentials import reject_credential_kwargs, vault_mode

                if vault_mode():
                    reject_credential_kwargs(embedding_kwargs)
                    _embeddings = _rotating_embeddings(
                        lambda key: OpenAIEmbeddings(model=model, openai_api_key=key, **embedding_kwargs))
                else:
                    _embeddings = OpenAIEmbeddings(model=model, **embedding_kwargs)
            case "azure_openai":
                from langchain_openai import AzureOpenAIEmbeddings

                _embeddings = AzureOpenAIEmbeddings(
                    model=model,
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                    **embedding_kwargs,
                )
            case "cohere":
                from langchain_cohere import CohereEmbeddings

                _embeddings = CohereEmbeddings(model=model, **embedding_kwargs)
            case "google_vertexai":
                from langchain_google_vertexai import VertexAIEmbeddings

                _embeddings = VertexAIEmbeddings(model=model, **embedding_kwargs)
            case "google_genai":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                _embeddings = GoogleGenerativeAIEmbeddings(
                    model=model, **embedding_kwargs
                )
            case "fireworks":
                from langchain_fireworks import FireworksEmbeddings

                _embeddings = FireworksEmbeddings(model=model, **embedding_kwargs)
            case "gigachat":
                from langchain_gigachat import GigaChatEmbeddings

                _embeddings = GigaChatEmbeddings(model=model, **embedding_kwargs)
            case "ollama":
                from langchain_ollama import OllamaEmbeddings

                _embeddings = OllamaEmbeddings(
                    model=model,
                    base_url=os.environ["OLLAMA_BASE_URL"],
                    **embedding_kwargs,
                )
            case "together":
                from langchain_together import TogetherEmbeddings

                _embeddings = TogetherEmbeddings(model=model, **embedding_kwargs)
            case "netmind":
                from langchain_netmind import NetmindEmbeddings

                _embeddings = NetmindEmbeddings(model=model, **embedding_kwargs)
            case "mistralai":
                from langchain_mistralai import MistralAIEmbeddings

                _embeddings = MistralAIEmbeddings(model=model, **embedding_kwargs)
            case "huggingface":
                from langchain_huggingface import HuggingFaceEmbeddings

                _embeddings = HuggingFaceEmbeddings(model_name=model, **embedding_kwargs)
            case "nomic":
                from langchain_nomic import NomicEmbeddings

                _embeddings = NomicEmbeddings(model=model, **embedding_kwargs)
            case "voyageai":
                from langchain_voyageai import VoyageAIEmbeddings

                _embeddings = VoyageAIEmbeddings(
                    voyage_api_key=os.environ["VOYAGE_API_KEY"],
                    model=model,
                    **embedding_kwargs,
                )
            case "dashscope":
                from langchain_community.embeddings import DashScopeEmbeddings

                _embeddings = DashScopeEmbeddings(model=model, **embedding_kwargs)
            case "bedrock":
                from langchain_aws.embeddings import BedrockEmbeddings

                _embeddings = BedrockEmbeddings(model_id=model, **embedding_kwargs)
            case "aimlapi":
                from langchain_openai import OpenAIEmbeddings

                _embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=os.getenv("AIMLAPI_API_KEY"),
                    openai_api_base=os.getenv("AIMLAPI_BASE_URL", "https://api.aimlapi.com/v1"),
                    **embedding_kwargs,
                )
            case _:
                raise Exception("Embedding not found.")

        self._embeddings = _embeddings

    def get_embeddings(self):
        return self._embeddings


_ROTATING_EMBEDDINGS = None


def _rotating_embeddings(factory):
    """Vault mode: OpenAI embeddings that read the current key at every embedding call.

    The wrapped LangChain client is rebuilt only when the accepted key changed, so budget metering clients, chunk
    size and retry settings in the factory are preserved. A call already in flight finishes with its client; an
    unavailable or revoked key fails the next call instead of reusing the old client.
    """
    global _ROTATING_EMBEDDINGS
    if _ROTATING_EMBEDDINGS is None:
        import threading
        from langchain_core.embeddings import Embeddings
        from gpt_researcher.credentials import provider_credential

        class RotatingOpenAIEmbeddings(Embeddings):
            def __init__(self, build):
                self._build = build
                self._lock = threading.Lock()
                self._generation = None
                self._client = None

            def __repr__(self):
                return "RotatingOpenAIEmbeddings()"

            def _current(self):
                credential = provider_credential("openai-chat")
                with self._lock:
                    if self._client is None or self._generation != credential.generation:
                        self._client = self._build(credential.value)
                        self._generation = credential.generation
                    return self._client

            def embed_documents(self, texts):
                return self._current().embed_documents(texts)

            def embed_query(self, text):
                return self._current().embed_query(text)

            async def aembed_documents(self, texts):
                return await self._current().aembed_documents(texts)

            async def aembed_query(self, text):
                return await self._current().aembed_query(text)

        _ROTATING_EMBEDDINGS = RotatingOpenAIEmbeddings
    return _ROTATING_EMBEDDINGS(factory)
