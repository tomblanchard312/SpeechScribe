"""
Control Plane - Environment Detection

Environment awareness for deployment contexts.
"""

import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments."""
    AZURE = "azure"
    OFFLINE = "offline"
    LOCAL = "local"


class EnvironmentDetector:
    """
    Detects the current deployment environment.
    """

    @staticmethod
    def detect() -> Environment:
        """Detect the current deployment environment."""
        # Check for Azure environment variables
        if os.getenv('AZURE_ENVIRONMENT') or os.getenv('WEBSITE_INSTANCE_ID'):
            return Environment.AZURE

        # Check for offline indicators (no internet, etc.)
        # For now, assume local unless Azure detected
        return Environment.LOCAL

    @staticmethod
    def is_offline_mode() -> bool:
        """Check if running in offline mode."""
        return EnvironmentDetector.detect() == Environment.OFFLINE

    @staticmethod
    def is_azure_mode() -> bool:
        """Check if running in Azure."""
        return EnvironmentDetector.detect() == Environment.AZURE

    @staticmethod
    def is_local_mode() -> bool:
        """Check if running locally."""
        return EnvironmentDetector.detect() == Environment.LOCAL
