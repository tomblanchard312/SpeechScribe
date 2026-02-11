"""
CLI utilities: secure error handling and common helpers.

Prevents leakage of sensitive system information through error messages
while preserving full detail in logs for debugging.
"""

import logging
import sys

import click


def handle_cli_error(
    logger: logging.Logger,
    user_message: str,
    exc: BaseException,
    exit_code: int = 1,
) -> None:
    """
    Handle an exception in a CLI command securely.

    Logs the full exception (including traceback) for debugging.
    Shows only a generic user-facing message to avoid leaking internal details.
    Then exits with the given code.
    """
    logger.exception("CLI error: %s", exc)
    click.echo(user_message, err=True)
    sys.exit(exit_code)
