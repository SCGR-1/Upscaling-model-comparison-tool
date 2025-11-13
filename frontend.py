"""Frontend HTML template."""
from pathlib import Path

def get_frontend_html() -> str:
    """Load and return the frontend HTML template."""
    html_path = Path(__file__).parent / "frontend.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    # Fallback: return empty if file doesn't exist
    return "<html><body><h1>Frontend HTML file not found</h1></body></html>"

