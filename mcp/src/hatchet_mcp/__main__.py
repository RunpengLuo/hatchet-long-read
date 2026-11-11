"""Entry point: ``hatchet-mcp`` or ``python -m hatchet_mcp`` starts the stdio server."""

from hatchet_mcp.server import mcp


def main():
    """Run the MCP server over stdio."""
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
