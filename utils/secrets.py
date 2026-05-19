"""Load optional API keys from environment (never commit real keys)."""
import os


def get_api_key(name: str, default: str = 'Your_Key') -> str:
    """Return env var ``name`` or *default* placeholder."""
    return os.environ.get(name, default).strip()


def is_placeholder(value: str) -> bool:
    if not value:
        return True
    v = value.strip().lower()
    return v in ('your_key', 'your-key', 'changeme', 'none', 'null', '')


def load_dotenv(path: str = '.env') -> None:
    """Load KEY=VALUE lines from ``.env`` without extra dependencies."""
    if not os.path.isfile(path):
        return
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def configure_wandb_login():
    """
    Log in to Weights & Biases when a real key is set.

    Set ``WANDB_API_KEY=Your_Key`` in ``.env.example`` locally; export a real key only on your machine.
    Returns wandb ``anonymous`` mode for ``wandb.init``: None (logged in) or ``'allow'`` (offline / anonymous).
    """
    import wandb

    key = get_api_key('WANDB_API_KEY')
    if not is_placeholder(key):
        wandb.login(key=key)
        return None
    return 'allow'
