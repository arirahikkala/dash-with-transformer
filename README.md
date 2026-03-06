# Dash with Transformer

An information-theory-based text input method powered by a language model. It displays a widget that maps a language model's probability distribution onto a zoomable 2D space: you glide rightward to write, up/down to choose content, and leftward to correct mistakes.

This project is unaffiliated with [Dasher](https://github.com/dasher-project), but is obviously very directly inspired by it.

The front-end defaults to a small in-browser CPU LSTM model trained on FineWeb, which should somewhat work. You may also set up the backend, which uses vLLM and seems to work based on casual testing.

## Architecture

- **Frontend** (`frontend/`) — Vite + TypeScript.
- **Backend** (`backend/`) — Python + FastAPI.

## Frontend

### Setup

```bash
cd frontend
npm install
npm run dev
```

### Configuration

Configuration is passed via URL hash parameters, e.g. `http://localhost:5173/#backendUrl=http://myhost:8000&remoteModelCallPrefix=Once upon a time`.

| Parameter | Default | Description |
|---|---|---|
| `backendUrl` | `http://localhost:8000` | URL of the backend server |
| `remoteModelCallPrefix` | (empty) | Text prefix invisibly prepended to remote model calls. Useful for models that just predict endoftext off a BOS-only context, for instance. |

## Backend

### Setup

```bash
cd backend
pip install -e .
```

### Running

```bash
dash-with-transformer --host 0.0.0.0 --port 8000
```

Or equivalently: `python -m dash_with_transformer.server`

#### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |

#### Environment variables

Configured in `engine.py` and read at import time:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `gpt2-medium` | Huggingface model ID |
| `BEAM_K` | `5` | Beam width for `ByteBeamState` inference |
| `PRUNE_THRESHOLD` | `0.05` | Beam pruning threshold — beams with probability below this fraction of the best beam are dropped |
| `CACHE_MAX_SIZE` | `10000` | Maximum number of entries in the LRU state cache |
| `INITIAL_CONTEXT` | (none) | Comma-separated token IDs fed as initial context before inference. |
| `MAX_MODEL_LEN` | (none) | Maximum sequence length, passed as max_model_len to vLLM. |
| `TRIE_TIME_LIMIT` | `1.0` | Time limit in seconds for trie expansion per request. When exceeded, the server returns whatever trie it has built so far. |

## Tests

```bash
cd frontend
npm test
```

## Compatibility notes

Qwen models don't set a BOS token ID, and our custom version of genlm-bytes currently sets an empty initial context if there's no BOS token, so you need to provide an INITIAL_CONTEXT. A working substitute is a single EOS token, so, for Qwen 2.5 and 3, `INITIAL_CONTEXT=151643`. For Qwen 3.5, you might try `INITIAL_CONTEXT=248044`, but I haven't successfully tested that yet.