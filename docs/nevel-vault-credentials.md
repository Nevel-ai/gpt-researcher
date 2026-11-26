# Nevel deployment: provider credentials from Vault

Applies to the Nevel fork only. Code: `gpt_researcher/credentials/`. Tests: `tests/test_research_credentials.py`.

## Mode switch

`CHAT_CONFIG_CREDENTIAL_SOURCE` (the same switch as Chat's Node processes), read once per worker process:

| Value | Behaviour |
|---|---|
| unset or `env` | Upstream behaviour. Every call site reads its env variable as before; no Vault client, thread or request. Request headers may still carry `tavily_api_key`/`google_api_key` as upstream allows. |
| `vault` | `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `TAVILY_API_KEY` come only from Vault. Env copies are ignored and never a fallback. Request headers or configured client kwargs that carry a credential are rejected (`credential_override_forbidden`). The `custom` embedding provider is refused because it would read the env copy. |
| anything else | The worker fails at startup. |

There is no automatic switching in either direction. Manual emergency mode: set the variable to `env` on the
Deployment (rolling restart); the SOPS env copies remain delivered for this.

Vault mode also needs `CHAT_CONFIG_VAULT_ENV` (`dev` or `prod`; there is no staging Researcher, previews use dev)
and `CHAT_CONFIG_VAULT_ROLE`. Fixed in code, as in Chat's `vaultClientOptions`: address `https://vault.nevel.ai`,
auth mount `eks-<env>`, projected ServiceAccount token `/var/run/nevel-vault/token`. The token file is reread at
every login.

Documents (existing provider documents shared with Chat; one field each):

| Source | KV path (mount `kv`) | Field |
|---|---|---|
| `openrouter-chat` | `providers/openrouter/<env>/chat` | `OPENROUTER_API_KEY` |
| `openai-chat` | `providers/openai/<env>/chat` | `OPENAI_API_KEY` |
| `tavily-chat` | `providers/tavily/<env>/chat` | `TAVILY_API_KEY` |

The role needs exactly `read` on these three `kv/data/...` paths and `update` on `auth/token/renew-self`.

## Cache, refresh and revocation

- One background thread per worker process polls every 15 s (+/-20%), backing off 1 s → 60 s after failures.
  Business requests never wait for Vault.
- A validated value stays usable for 48 hours after its last successful read, in memory only. A restart starts
  empty. Login, renewal and failed reads do not extend it.
- Network errors, timeouts, 429/5xx and responses that are not shaped like Vault's keep the cache.
- Vault's 404 or a deleted/destroyed version closes that source at once. A 403 on a read triggers one new login
  and a repeat; a 403 with the fresh token closes that source. A denied login closes all sources. A Kubernetes
  JWKS failure looks the same to Vault as a revocation.
- A new version that fails validation is ignored; the accepted one keeps its original age. A value must be
  printable ASCII without spaces or control characters (it travels in HTTP headers and JSON bodies).
- The failure backoff is capped before it is computed, so an outage of any length keeps polling every 60 s.
- The token is renewed at 60–70% of its TTL; a denied renewal, expiry or a lease capped by `token_max_ttl`
  leads to a new login.
- HTTPS with certificate verification, no redirects, env proxies ignored, 5 s per request, responses limited to
  1 MiB while reading. Logs carry only source, code, status and a short reason.

Not implemented (not needed for these documents): `delete_version_after` deadlines, subscriptions, AppRole.

## When a new key is used

The latest validated key is taken when a provider client is built, not on every HTTP request:

| Operation | Client lifetime | Effect of a rotation |
|---|---|---|
| OpenRouter / OpenAI chat completion | new `ChatOpenAI` per completion | next completion uses the new key |
| Report chat with tools (`/api/chat`, `/api/reports/{id}/chat`) | tool-bound client checked before each paid invocation | the follow-up after tools is rebound to a rotated key; a withdrawn key ends the request before the follow-up, with no fallback completion |
| Tavily search | key resolved for every search request | next search request |
| Tavily extract, report chat search | client built per extraction / search | next extraction / search |
| OpenAI embeddings | wrapper checks the key at every embedding call, rebuilds its client only when it changed | next embedding call |

A request or stream already running finishes with the key it started with. Nothing is closed, retried or
re-sent with the new key, so no paid request is duplicated. During a rotation keep the old provider key valid
until running requests can no longer use it: at least the longest research run (Chat's deep research ceiling
defaults to 10 minutes) plus the polling window. Then revoke it at the provider.

After a revocation or deletion, the next client construction / embedding call fails with
`credentials_unavailable`; a request already in flight is not interrupted.

## Readiness and entrypoints

- `GET /health`: liveness, independent of credentials.
- `GET /ready`: 503 until all three sources are usable in this worker; body is only
  `{"credentials": {"mode", "ready"}}`. With several uvicorn workers a probe reaches one of them, so every
  business entrypoint also checks its own worker before starting.
- Refused before budget scoping or any paid call when credentials are unavailable: WebSocket `start` on `/ws` and
  `/ws/budget-v2` (after the existing budget capability check, which still closes invalid starts with 1008),
  `/report/`, `/api/multi_agents`, `/api/chat`, `/api/reports/{id}/chat`, and `run_agent` itself. Credentials
  lost during a run fail the run (`credentials_unavailable`) even where legacy code swallows provider errors.
  Tool errors and the tool helper's fallback completion never absorb a credential failure.
- WebSocket refusal: `{"type": "error", "output": ..., "metadata": {"credentials_error_code": ...}}`; HTTP: 503
  (unavailable) or 400 (override), `{"error", "code"}`.

Not managed here: other retrievers/LLM providers, `NEVEL_BUDGET_PUBLIC_KEY` (public), `JWT_SECRET` (no reader
since the Ed25519 budget capabilities; its env delivery is kept for rollback to older images).
