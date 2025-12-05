import os
from unittest.mock import patch
import pytest

from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.llms.providers.anthropic.completion import AnthropicCompletion


class TestGPUAIIntegration:
    """Tests for GPU AI integration feature"""

    def test_openai_uses_default_url_without_gpuai(self):
        """Test that OpenAI uses default URL when USE_GPUAI is not set"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            completion = OpenAICompletion(model="gpt-4o")
            client_params = completion._get_client_params()
            
            # Should not have base_url set (defaults to OpenAI)
            assert client_params.get("base_url") is None

    def test_openai_uses_gpuai_when_enabled(self):
        """Test that OpenAI uses GPU AI URL when USE_GPUAI is true"""
        with patch.dict(
            os.environ, 
            {"OPENAI_API_KEY": "test-key", "USE_GPUAI": "true"}, 
            clear=True
        ):
            completion = OpenAICompletion(model="gpt-4o")
            client_params = completion._get_client_params()
            
            assert client_params["base_url"] == "https://gpuai.app/api/v1"

    def test_openai_uses_gpuai_with_various_true_values(self):
        """Test that USE_GPUAI works with different truthy values"""
        test_values = ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]
        
        for value in test_values:
            with patch.dict(
                os.environ, 
                {"OPENAI_API_KEY": "test-key", "USE_GPUAI": value}, 
                clear=True
            ):
                completion = OpenAICompletion(model="gpt-4o")
                client_params = completion._get_client_params()
                
                assert client_params["base_url"] == "https://gpuai.app/api/v1", \
                    f"Failed for USE_GPUAI={value}"

    def test_openai_ignores_gpuai_with_false_values(self):
        """Test that USE_GPUAI is ignored when set to false"""
        test_values = ["false", "False", "FALSE", "0", "no", "No", "NO"]
        
        for value in test_values:
            with patch.dict(
                os.environ, 
                {"OPENAI_API_KEY": "test-key", "USE_GPUAI": value}, 
                clear=True
            ):
                completion = OpenAICompletion(model="gpt-4o")
                client_params = completion._get_client_params()
                
                assert client_params.get("base_url") is None, \
                    f"Failed for USE_GPUAI={value}"

    def test_openai_explicit_base_url_takes_precedence_over_gpuai(self):
        """Test that explicitly set base_url takes precedence over USE_GPUAI"""
        with patch.dict(
            os.environ, 
            {"OPENAI_API_KEY": "test-key", "USE_GPUAI": "true"}, 
            clear=True
        ):
            custom_url = "https://custom-api.example.com/v1"
            completion = OpenAICompletion(model="gpt-4o", base_url=custom_url)
            client_params = completion._get_client_params()
            
            # Custom base_url should take precedence
            assert client_params["base_url"] == custom_url

    def test_openai_env_base_url_takes_precedence_over_gpuai(self):
        """Test that OPENAI_BASE_URL env var takes precedence over USE_GPUAI"""
        custom_url = "https://custom-api.example.com/v1"
        with patch.dict(
            os.environ, 
            {
                "OPENAI_API_KEY": "test-key", 
                "USE_GPUAI": "true",
                "OPENAI_BASE_URL": custom_url
            }, 
            clear=True
        ):
            completion = OpenAICompletion(model="gpt-4o")
            client_params = completion._get_client_params()
            
            # OPENAI_BASE_URL should take precedence
            assert client_params["base_url"] == custom_url

    def test_anthropic_uses_default_url_without_gpuai(self):
        """Test that Anthropic uses default URL when USE_GPUAI is not set"""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            completion = AnthropicCompletion(model="claude-3-5-sonnet-20241022")
            client_params = completion._get_client_params()
            
            # Should not have base_url set (defaults to Anthropic)
            assert client_params.get("base_url") is None

    def test_anthropic_uses_gpuai_when_enabled(self):
        """Test that Anthropic uses GPU AI URL when USE_GPUAI is true"""
        with patch.dict(
            os.environ, 
            {"ANTHROPIC_API_KEY": "test-key", "USE_GPUAI": "true"}, 
            clear=True
        ):
            completion = AnthropicCompletion(model="claude-3-5-sonnet-20241022")
            client_params = completion._get_client_params()
            
            assert client_params["base_url"] == "https://gpuai.app/api/v1"

    def test_anthropic_uses_gpuai_with_various_true_values(self):
        """Test that USE_GPUAI works with different truthy values for Anthropic"""
        test_values = ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]
        
        for value in test_values:
            with patch.dict(
                os.environ, 
                {"ANTHROPIC_API_KEY": "test-key", "USE_GPUAI": value}, 
                clear=True
            ):
                completion = AnthropicCompletion(model="claude-3-5-sonnet-20241022")
                client_params = completion._get_client_params()
                
                assert client_params["base_url"] == "https://gpuai.app/api/v1", \
                    f"Failed for USE_GPUAI={value}"

    def test_anthropic_ignores_gpuai_with_false_values(self):
        """Test that USE_GPUAI is ignored when set to false for Anthropic"""
        test_values = ["false", "False", "FALSE", "0", "no", "No", "NO"]
        
        for value in test_values:
            with patch.dict(
                os.environ, 
                {"ANTHROPIC_API_KEY": "test-key", "USE_GPUAI": value}, 
                clear=True
            ):
                completion = AnthropicCompletion(model="claude-3-5-sonnet-20241022")
                client_params = completion._get_client_params()
                
                assert client_params.get("base_url") is None, \
                    f"Failed for USE_GPUAI={value}"

    def test_anthropic_explicit_base_url_takes_precedence_over_gpuai(self):
        """Test that explicitly set base_url takes precedence over USE_GPUAI for Anthropic"""
        with patch.dict(
            os.environ, 
            {"ANTHROPIC_API_KEY": "test-key", "USE_GPUAI": "true"}, 
            clear=True
        ):
            custom_url = "https://custom-api.example.com/v1"
            completion = AnthropicCompletion(
                model="claude-3-5-sonnet-20241022", 
                base_url=custom_url
            )
            client_params = completion._get_client_params()
            
            # Custom base_url should take precedence
            assert client_params["base_url"] == custom_url
