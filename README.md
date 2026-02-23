# Dash with Transformer

An information-theory-based text input method powered by a language model. It displays a widget that maps a language model's probability distribution onto a zoomable 2D space: you glide rightward to write, up/down to choose content, and leftward to correct mistakes.

This project is unaffiliated with [Dasher](https://en.wikipedia.org/wiki/Dasher_(software)), but is obviously very directly inspired by it.

## Architecture

- **Frontend** (`frontend/`) — Vite + TypeScript. Renders the interactive widget on a canvas and drives mouse-based navigation through the language model's probability space.
- **Backend** (`backend/`) — Python + FastAPI. Wraps a byte-level language model (via `genlm-bytes`) and serves next-byte probability distributions over a `/predict` endpoint.

## Frontend

### Setup

```bash
cd frontend
npm install
npm run dev
```

### Configuration

Configuration is passed via URL hash parameters, e.g. `http://localhost:5173/#backendUrl=http://myhost:8000&modelCallPrefix=Once upon a time`.

| Parameter | Default | Description |
|---|---|---|
| `backendUrl` | `http://localhost:8000` | URL of the backend server |
| `modelCallPrefix` | (empty) | Text prefix invisibly prepended to every model call, letting you steer generation from a given context |

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
| `MODEL_NAME` | `gpt2-medium` | Model name passed to `genlm`'s `load_model_by_name` |
| `BEAM_K` | `5` | Beam width for `ByteBeamState` inference |
| `PRUNE_THRESHOLD` | `0.05` | Beam pruning threshold — beams with probability below this fraction of the best beam are dropped |
| `CACHE_MAX_SIZE` | `10000` | Maximum number of entries in the LRU state cache |

## Tests

```bash
cd frontend
npm test
```
