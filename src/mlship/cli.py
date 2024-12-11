import click
import os
import sys
import logging
from pathlib import Path
from .server import start_server

logger = logging.getLogger(__name__)

@click.group()
def cli():
    """MLship - Simple ML Model Deployment"""
    pass

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--ui/--no-ui', default=True, help='Enable/disable web interface')
def deploy(model_path, host, port, ui):
    """Deploy a model as a REST API."""
    try:
        # Convert model path to absolute path if relative
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
            
        click.echo(f"🚀 API: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        if ui:
            click.echo(f"🎨 UI: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/ui")
        else:
            click.echo("UI disabled")
            
        start_server(model_path, host=host, port=port, ui=ui)
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        click.echo(f"Failed to load model: {str(e)}")
        sys.exit(1)

def main():
    cli()