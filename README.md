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

## Tests

```bash
cd frontend
npm test
```
