import os
import requests
from typing import Optional, Dict, Any
from urllib.parse import urljoin

class LLMManager:

    """Manager for interacting with LLM API endpoints"""
    
    def __init__(self, base_url: str = "http://host.docker.internal:25000"):
        """
        Initialize LLM manager
        
        Args:
            base_url: Base URL of LLM API server
        
        Raises:
            ValueError: If LLM_AUTH_KEY environment variable is not set
        """
        self.base_url = base_url.rstrip('/')
        self.auth_key = os.getenv("LLM_AUTH_KEY")
        if not self.auth_key:
            raise ValueError("LLM_AUTH_KEY environment variable not set")
        self.current_key: Optional[str] = None
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request to API endpoint
        
        Args:
            method: HTTP method (get, post, etc)
            endpoint: API endpoint path
            **kwargs: Additional arguments passed to requests
            
        Returns:
            Dict containing API response
            
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        url = urljoin(f"{self.base_url}/", endpoint.lstrip('/'))
        headers = kwargs.pop('headers', {})
        headers['Authorization'] = self.auth_key
        
        response = requests.request(
            method,
            url,
            headers=headers,
            **kwargs
        )
        response.raise_for_status()
        return response.json()

    def init_key(self, key: str) -> Dict[str, str]:
        """
        Initialize token tracking for a key
        
        Args:
            key: Key to initialize
            
        Returns:
            Dict containing initialization status
        """
        result = self._make_request(
            'post',
            'init',
            json={'key': key}
        )
        self.current_key = key
        return result

    def reset_count(self) -> Dict[str, str]:
        """
        Reset token count for current key
        
        Returns:
            Dict containing reset status
        """
        return self._make_request('post', 'reset')

    def get_count(self) -> Dict[str, Any]:
        """
        Get current token count
        
        Returns:
            Dict containing current key and count
        """
        return self._make_request('get', 'count')
