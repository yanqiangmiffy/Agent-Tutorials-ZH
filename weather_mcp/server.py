import json
import requests
import logging
from mcp.server.fastmcp import FastMCP
logger = logging.getLogger(__name__)

# Create MCP server instance
mcp = FastMCP(
    name="WeatherForecastServer",
    description="Provides global weather forecasts and current weather conditions using wttr.in service",
)
def get_current_weather(city: str) -> str:
    """
    Get current weather for a specified location using wttr.in service.

    Parameters:
        city: Location name, e.g., "Beijing", "New York", "Tokyo", "武汉"

    Returns:
        str: Current weather information in plain text format.
    """
    logger.debug(f"get_current_weather({city})")
    try:
        endpoint = "https://wttr.in"
        # Get text format weather data
        response = requests.get(f"{endpoint}/{city}")
        response.raise_for_status()
        text_result = response.text
        logger.debug(f"Weather data for {city}: {text_result}")
        return text_result
    except Exception as e:
        logger.error(f"Error in getting weather for {city}: {str(e)}")
        return json.dumps({"operation": "get_current_weather", "error": str(e)})



@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get current weather for a specified location using wttr.in service.

    Parameters:
        city: Location name, e.g., "Beijing", "New York", "Tokyo", "武汉"

    Returns:
        str: Current weather information in plain text format.
    """
    return get_current_weather(city)


def run_server(transport="stdio"):
    """
    Run the MCP server with the specified transport.

    Args:
        transport: The transport type to use ('stdio', 'sse', etc.)
    """
    logger.info(f"Starting Weather Forecast MCP Server...")
    mcp.run(transport=transport)  # noqa

if __name__ == "__main__":
    mcp.run()

