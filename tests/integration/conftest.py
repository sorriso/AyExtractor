# tests/integration/conftest.py — v7
"""Shared fixtures for integration tests using testcontainers.

Container lifecycle:
- session scope: containers start once per pytest session
- function scope: fresh collection/database per test for isolation

Custom container wrappers:
- Uses DockerContainer directly with bridge network IP + internal port
- Required for devcontainer with docker-outside-of-docker (socket mount)
- Built-in testcontainers library returns localhost:mapped_port which is
  unreachable from inside a devcontainer

Changelog:
    v7: Fix Qdrant wait_for_logs regex ("Actix runtime" not "starting N worker").
        Fix ChromaDB wait_for_logs regex ("######" banner for chroma 1.0.0).
        Fix exec_run: remove unsupported 'timeout' kwarg from docker-py.
    v6: Bridge IP pattern for all containers.
    v5: MockEmbedder + model_name, chromadb http-only guard.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from typing import Any

import pytest
from pydantic import BaseModel

from ayextractor.llm.base_client import BaseLLMClient
from ayextractor.llm.models import LLMResponse, Message

logger = logging.getLogger(__name__)


# ── Pytest markers ──────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that require model download (Ollama)")
    config.addinivalue_line("markers", "arangodb: marks tests requiring ArangoDB container")
    config.addinivalue_line("markers", "qdrant: marks tests requiring Qdrant container")
    config.addinivalue_line("markers", "chromadb: marks tests requiring ChromaDB container")
    config.addinivalue_line("markers", "ollama: marks tests requiring Ollama container")
    config.addinivalue_line("markers", "neo4j: marks tests requiring Neo4j container")
    config.addinivalue_line("markers", "redis: marks tests requiring Redis container")


# =====================================================================
#  DEVCONTAINER NETWORKING HELPERS
# =====================================================================

def _get_container_bridge_ip(container, max_attempts: int = 10) -> str:
    """Get container bridge network IP with retries.

    In docker-outside-of-docker setups, containers are on the host Docker
    daemon. The devcontainer must access them via bridge IP, not localhost.
    """
    for attempt in range(max_attempts):
        try:
            wrapped = container.get_wrapped_container()
            wrapped.reload()
            networks = wrapped.attrs.get("NetworkSettings", {}).get("Networks", {})
            for net_name, net_info in networks.items():
                ip = net_info.get("IPAddress", "")
                if ip:
                    logger.info(
                        "Container %s IP: %s (network: %s, attempt %d)",
                        wrapped.short_id, ip, net_name, attempt + 1,
                    )
                    return ip
            logger.debug("Container IP empty, attempt %d/%d", attempt + 1, max_attempts)
        except Exception as e:
            logger.debug("Error getting IP (attempt %d): %s", attempt + 1, e)
        time.sleep(0.5)

    raise RuntimeError(
        f"Could not obtain container bridge IP after {max_attempts} attempts"
    )


def _docker_available() -> bool:
    """Check if Docker daemon is reachable."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


skip_no_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available",
)


# =====================================================================
#  MOCK LLM CLIENT — no Docker required
# =====================================================================

class MockLLMClient(BaseLLMClient):
    """Mock LLM client for integration testing without real LLM services.

    Implements BaseLLMClient abstract interface:
    - complete(), complete_with_vision(), supports_vision, provider_name
    """

    def __init__(self, default_response: str = '{"result": "mock"}'):
        self._default_response = default_response
        self._response_queue: list[str] = []
        self.calls: list[dict[str, Any]] = []

    def set_responses(self, *responses: str) -> None:
        self._response_queue = list(responses)

    def set_default(self, response: str) -> None:
        self._default_response = response

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        content = self._response_queue.pop(0) if self._response_queue else self._default_response
        self.calls.append({
            "messages": messages, "system": system,
            "max_tokens": max_tokens, "temperature": temperature,
            "response_format": response_format,
        })
        return LLMResponse(
            content=content, input_tokens=50, output_tokens=len(content) // 4,
            model="mock-model", provider="mock", latency_ms=10,
            raw_response={"mock": True},
        )

    async def complete_with_vision(
        self,
        messages: list[Message],
        images: list,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        content = self._response_queue.pop(0) if self._response_queue else self._default_response
        self.calls.append({"messages": messages, "images": images, "system": system})
        return LLMResponse(
            content=content, input_tokens=100, output_tokens=len(content) // 4,
            model="mock-model", provider="mock", latency_ms=15,
            raw_response={"mock": True},
        )

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "mock"


class MockEmbedder:
    """Deterministic embedder for testing — hashes text to produce vectors.

    Implements full BaseEmbedder interface:
    embed_texts(), embed_query(), dimensions, provider_name, model_name
    """

    def __init__(self, dimensions: int = 128):
        self._dims = dimensions
        self.call_count = 0

    def _text_to_vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).hexdigest()
        raw = [int(digest[i:i+2], 16) / 255.0 - 0.5 for i in range(0, min(len(digest), self._dims * 2), 2)]
        while len(raw) < self._dims:
            raw.extend(raw[:self._dims - len(raw)])
        raw = raw[:self._dims]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += len(texts)
        return [self._text_to_vec(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        self.call_count += 1
        return self._text_to_vec(query)

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-embedder"


@pytest.fixture
def mock_llm():
    return MockLLMClient()


@pytest.fixture
def mock_embedder():
    return MockEmbedder(dimensions=128)


# =====================================================================
#  IN-MEMORY STORES — no Docker required
# =====================================================================

@pytest.fixture
def qdrant_memory_store():
    from ayextractor.rag.vector_store.qdrant_store import QdrantStore
    return QdrantStore()


@pytest.fixture
def chromadb_memory_store():
    from ayextractor.rag.vector_store.chromadb_store import ChromaDBStore
    try:
        return ChromaDBStore()
    except RuntimeError as e:
        if "http-only" in str(e).lower():
            pytest.skip("chromadb http-only client; install full: pip install chromadb")
        raise


@pytest.fixture
def unique_collection() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


# =====================================================================
#  ARANGODB CONTAINER — session scope (bridge IP)
# =====================================================================

ARANGO_IMAGE = "arangodb:3.12"
ARANGO_INTERNAL_PORT = 8529
ARANGO_ROOT_PASSWORD = "testpassword"


@pytest.fixture(scope="session")
def arangodb_container():
    if not _docker_available():
        pytest.skip("Docker not available")

    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    container = (
        DockerContainer(ARANGO_IMAGE)
        .with_exposed_ports(ARANGO_INTERNAL_PORT)
        .with_env("ARANGO_ROOT_PASSWORD", ARANGO_ROOT_PASSWORD)
        .with_env("ARANGO_NO_AUTH", "0")
    )
    container.start()
    wait_for_logs(container, predicate=r"ArangoDB.*is ready for business", timeout=60)
    time.sleep(2)

    ip = _get_container_bridge_ip(container)
    logger.info("ArangoDB ready at %s:%d", ip, ARANGO_INTERNAL_PORT)
    yield {"host": ip, "port": ARANGO_INTERNAL_PORT, "password": ARANGO_ROOT_PASSWORD}
    container.stop()


@pytest.fixture(scope="session")
def arangodb_url(arangodb_container) -> str:
    c = arangodb_container
    return f"http://{c['host']}:{c['port']}"


@pytest.fixture
def arangodb_store(arangodb_url):
    from ayextractor.rag.graph_store.arangodb_store import ArangoDBStore
    suffix = uuid.uuid4().hex[:8]
    store = ArangoDBStore(
        url=arangodb_url, database="_system", user="root", password=ARANGO_ROOT_PASSWORD,
        graph_name=f"test_g_{suffix}", node_collection=f"test_n_{suffix}",
        edge_collection=f"test_e_{suffix}",
    )
    yield store
    try:
        db = store._db
        if db.has_graph(store._graph_name):
            db.delete_graph(store._graph_name, drop_collections=True)
        else:
            for col_name in [store._node_col, store._edge_col]:
                if db.has_collection(col_name):
                    db.delete_collection(col_name)
    except Exception:
        pass


# =====================================================================
#  QDRANT CONTAINER — session scope (bridge IP)
#
#  Qdrant v1.13.x logs: "... time found; starting in Actix runtime"
#  or "... gRPC listening on ..."
# =====================================================================

QDRANT_IMAGE = "qdrant/qdrant:v1.16"
QDRANT_HTTP_PORT = 6333
QDRANT_GRPC_PORT = 6334


@pytest.fixture(scope="session")
def qdrant_container():
    if not _docker_available():
        pytest.skip("Docker not available")

    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    container = (
        DockerContainer(QDRANT_IMAGE)
        .with_exposed_ports(QDRANT_HTTP_PORT, QDRANT_GRPC_PORT)
    )
    container.start()

    # Qdrant og format: "... time found; starting in Actix runtime"
    wait_for_logs(container, predicate=r"Actix runtime", timeout=60)
    time.sleep(1)

    ip = _get_container_bridge_ip(container)
    logger.info("Qdrant ready at %s:%d", ip, QDRANT_HTTP_PORT)
    yield {"host": ip, "port": QDRANT_HTTP_PORT}
    container.stop()


@pytest.fixture(scope="session")
def qdrant_url(qdrant_container) -> str:
    c = qdrant_container
    return f"http://{c['host']}:{c['port']}"


@pytest.fixture
def qdrant_store(qdrant_url):
    from ayextractor.rag.vector_store.qdrant_store import QdrantStore
    return QdrantStore(url=qdrant_url)


@pytest.fixture
def qdrant_collection() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


# =====================================================================
#  CHROMADB CONTAINER — session scope (bridge IP)
#
#  ChromaDB v1.0.0 logs: startup banner with "######" hash marks,
#  then eventually "Application startup complete" (uvicorn).
#  Use the broad banner pattern since uvicorn line may arrive late.
# =====================================================================

CHROMADB_IMAGE = "chromadb/chroma:1.5.0"
CHROMADB_INTERNAL_PORT = 8000


@pytest.fixture(scope="session")
def chromadb_container():
    if not _docker_available():
        pytest.skip("Docker not available")

    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    container = (
        DockerContainer(CHROMADB_IMAGE)
        .with_exposed_ports(CHROMADB_INTERNAL_PORT)
    )
    container.start()

    # ChromaDB : wait for hash banner then settle for port readiness
    wait_for_logs(container, predicate=r"######", timeout=120)
    # Allow extra settle time for uvicorn to bind
    time.sleep(3)

    ip = _get_container_bridge_ip(container)
    logger.info("ChromaDB ready at %s:%d", ip, CHROMADB_INTERNAL_PORT)
    yield {"host": ip, "port": CHROMADB_INTERNAL_PORT}
    container.stop()


@pytest.fixture(scope="session")
def chromadb_config(chromadb_container) -> dict:
    return chromadb_container


@pytest.fixture
def chromadb_store(chromadb_config):
    from ayextractor.rag.vector_store.chromadb_store import ChromaDBStore
    return ChromaDBStore(host=chromadb_config["host"], port=chromadb_config["port"])


@pytest.fixture
def chromadb_collection() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


# =====================================================================
#  OLLAMA CONTAINER — session scope (slow: model pull, bridge IP)
#
#  exec_run() in docker-py does NOT support 'timeout' kwarg.
#  Model pull is blocking; we just let it run.
# =====================================================================

OLLAMA_IMAGE = "ollama/ollama:latest"
OLLAMA_INTERNAL_PORT = 11434
OLLAMA_LLM_MODEL = "qwen2.5:0.5b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"


@pytest.fixture(scope="session")
def ollama_container():
    if not _docker_available():
        pytest.skip("Docker not available")

    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    container = (
        DockerContainer(OLLAMA_IMAGE)
        .with_exposed_ports(OLLAMA_INTERNAL_PORT)
    )
    container.start()

    wait_for_logs(container, predicate=r"Listening on", timeout=60)
    time.sleep(1)

    ip = _get_container_bridge_ip(container)
    endpoint = f"http://{ip}:{OLLAMA_INTERNAL_PORT}"
    logger.info("Ollama ready at %s", endpoint)

    # Pull models inside the container via exec_run (no timeout param in docker-py)
    logger.info("Pulling Ollama models (this may take several minutes)...")
    wrapped = container.get_wrapped_container()
    for model in [OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL]:
        logger.info("  Pulling %s ...", model)
        try:
            exit_code, output = wrapped.exec_run(f"ollama pull {model}")
            if exit_code != 0:
                logger.warning("Failed to pull %s: %s", model, output.decode(errors="replace"))
            else:
                logger.info("  Pulled %s OK", model)
        except Exception as e:
            logger.warning("Error pulling %s: %s", model, e)

    yield {"endpoint": endpoint, "host": ip, "port": OLLAMA_INTERNAL_PORT}
    container.stop()


@pytest.fixture(scope="session")
def ollama_endpoint(ollama_container) -> str:
    return ollama_container["endpoint"]


@pytest.fixture
def ollama_llm(ollama_endpoint):
    from ayextractor.llm.adapters.ollama_adapter import OllamaAdapter
    return OllamaAdapter(model=OLLAMA_LLM_MODEL, host=ollama_endpoint)


@pytest.fixture
def ollama_embedder(ollama_endpoint):
    from ayextractor.rag.embeddings.ollama_embedder import OllamaEmbedder
    return OllamaEmbedder(
        model=OLLAMA_EMBED_MODEL, base_url=ollama_endpoint, dimensions=768,
    )