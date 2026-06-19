"""
ngrok tunnel helper.

Usage (inside Kaggle / Colab kernel):

    from server.tunnel import start
    public_url = start(port=8000, token=os.environ["NGROK_TOKEN"])
    print("Connect your phone to:", public_url)

Requirements:
    pip install pyngrok

Getting a free ngrok token:
    1. Sign up at https://ngrok.com  (free, no credit card)
    2. Dashboard → Your Authtoken → copy the token
    3. On Kaggle: Add/Secrets → name=NGROK_TOKEN, value=<your token>
       On Colab:  Secrets (key icon) → name=NGROK_TOKEN, value=<your token>
"""
import os


def start(port: int = 8000, token: str = None) -> str:
    """
    Start an ngrok HTTPS tunnel on `port`.
    Returns the public URL (e.g. "https://abc123.ngrok-free.app").
    Raises RuntimeError if token is missing or tunnel fails.
    """
    try:
        from pyngrok import ngrok, conf as ngrok_conf
    except ImportError:
        raise RuntimeError("pyngrok not installed — run: pip install pyngrok")

    token = token or os.environ.get("NGROK_TOKEN") or os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        raise RuntimeError(
            "ngrok auth token not found.\n"
            "  Kaggle: Add → Secrets → NGROK_TOKEN = <your token>\n"
            "  Colab:  Secrets (key icon) → NGROK_TOKEN = <your token>\n"
            "  Local:  export NGROK_TOKEN=<your token>\n"
            "Get a free token at https://ngrok.com"
        )

    ngrok.set_auth_token(token)
    try:
        for t in ngrok.get_tunnels():
            ngrok.disconnect(t.public_url)
    except Exception:
        pass
    try:
        tunnel = ngrok.connect(port, "http", pooling_enabled=True)
    except Exception:
        # fallback without pooling (works when no stale tunnel exists)
        tunnel = ngrok.connect(port, "http")
    url = tunnel.public_url.replace("http://", "https://")
    print(f"\n{'='*60}")
    print(f"  BADMINTON SERVER LIVE")
    print(f"  Public URL : {url}")
    print(f"  Camera     : POST {url}/frame")
    print(f"  Display WS : {url.replace('https','wss')}/ws")
    print(f"  Status     : {url}/status")
    print(f"{'='*60}\n")
    return url
