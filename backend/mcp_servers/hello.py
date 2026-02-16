"""Hello World MCP Server for learning the MCP protocol.

This is a temporary server for understanding MCP concepts.
"""

from fastmcp import FastMCP

mcp = FastMCP("hello-server")


def greet_impl(name: str) -> str:
    """Greet a person by name.
    
    Args:
        name: The name of the person to greet
        
    Returns:
        A greeting message
    """
    return f"Hello, {name}!"


@mcp.tool()
def greet(name: str) -> str:
    """Greet a person by name.
    
    Args:
        name: The name of the person to greet
        
    Returns:
        A greeting message
    """
    return greet_impl(name)


if __name__ == "__main__":
    mcp.run()