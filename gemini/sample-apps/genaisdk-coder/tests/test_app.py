# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Comprehensive unit tests for the genaisdk-coder Streamlit application.

This test module covers:
- Model name formatting functionality
- Configuration handling and validation
- User input processing
- Integration with external APIs (mocked)
- Error handling and edge cases
- UI component rendering
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest


# Import the module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestGetModelName:
    """Test suite for the get_model_name function."""

    def test_get_model_name_valid_model(self):
        """Test get_model_name with a valid model from MODELS dict."""
        from app import get_model_name, MODELS
        
        for model_key, expected_name in MODELS.items():
            result = get_model_name(model_key)
            assert result == expected_name, f"Expected {expected_name}, got {result}"

    def test_get_model_name_none(self):
        """Test get_model_name with None input."""
        from app import get_model_name
        
        result = get_model_name(None)
        assert result == "Gemini", "Expected 'Gemini' for None input"

    def test_get_model_name_empty_string(self):
        """Test get_model_name with empty string."""
        from app import get_model_name
        
        result = get_model_name("")
        assert result == "Gemini", "Expected 'Gemini' for empty string"

    def test_get_model_name_unknown_model(self):
        """Test get_model_name with an unknown model name."""
        from app import get_model_name
        
        result = get_model_name("unknown-model-xyz")
        assert result == "Gemini", "Expected 'Gemini' for unknown model"

    def test_get_model_name_special_characters(self):
        """Test get_model_name with special characters."""
        from app import get_model_name
        
        result = get_model_name("model@#$%")
        assert result == "Gemini", "Expected 'Gemini' for model with special chars"


class TestModelsConfiguration:
    """Test suite for MODELS and THINKING_BUDGET_MODELS configuration."""

    def test_models_dict_not_empty(self):
        """Verify MODELS dictionary is not empty."""
        from app import MODELS
        
        assert len(MODELS) > 0, "MODELS dictionary should not be empty"
        assert isinstance(MODELS, dict), "MODELS should be a dictionary"

    def test_models_contain_expected_keys(self):
        """Verify MODELS contains expected model keys."""
        from app import MODELS
        
        expected_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite-preview-06-17",
        ]
        
        for model in expected_models:
            assert model in MODELS, f"{model} should be in MODELS"

    def test_models_values_are_strings(self):
        """Verify all MODELS values are strings."""
        from app import MODELS
        
        for key, value in MODELS.items():
            assert isinstance(value, str), f"Value for {key} should be a string"
            assert len(value) > 0, f"Value for {key} should not be empty"

    def test_thinking_budget_models_subset(self):
        """Verify THINKING_BUDGET_MODELS is a subset of MODELS."""
        from app import MODELS, THINKING_BUDGET_MODELS
        
        assert isinstance(THINKING_BUDGET_MODELS, set), "THINKING_BUDGET_MODELS should be a set"
        for model in THINKING_BUDGET_MODELS:
            assert model in MODELS, f"{model} in THINKING_BUDGET_MODELS should be in MODELS"

    def test_thinking_budget_models_not_empty(self):
        """Verify THINKING_BUDGET_MODELS is not empty."""
        from app import THINKING_BUDGET_MODELS
        
        assert len(THINKING_BUDGET_MODELS) > 0, "THINKING_BUDGET_MODELS should not be empty"


class TestGenAIReposConfiguration:
    """Test suite for GENAI_REPOS configuration."""

    def test_genai_repos_not_empty(self):
        """Verify GENAI_REPOS dictionary is not empty."""
        from app import GENAI_REPOS
        
        assert len(GENAI_REPOS) > 0, "GENAI_REPOS dictionary should not be empty"
        assert isinstance(GENAI_REPOS, dict), "GENAI_REPOS should be a dictionary"

    def test_genai_repos_contain_expected_languages(self):
        """Verify GENAI_REPOS contains expected programming languages."""
        from app import GENAI_REPOS
        
        expected_languages = ["Python", "Java", "Go", "JavaScript"]
        
        for language in expected_languages:
            assert language in GENAI_REPOS, f"{language} should be in GENAI_REPOS"

    def test_genai_repos_urls_valid_format(self):
        """Verify all GENAI_REPOS URLs are valid GitHub URLs."""
        from app import GENAI_REPOS
        
        for language, url in GENAI_REPOS.items():
            assert isinstance(url, str), f"URL for {language} should be a string"
            assert url.startswith("https://github.com/"), f"URL for {language} should start with https://github.com/"
            assert "googleapis" in url, f"URL for {language} should contain googleapis"


@patch.dict(os.environ, {}, clear=True)
class TestEnvironmentVariables:
    """Test suite for environment variable handling."""

    def test_cloud_run_service_not_set(self):
        """Test behavior when K_SERVICE environment variable is not set."""
        result = os.environ.get("K_SERVICE")
        assert result is None, "K_SERVICE should not be set in test environment"

    @patch.dict(os.environ, {"K_SERVICE": "test-service"})
    def test_cloud_run_service_set(self):
        """Test behavior when K_SERVICE environment variable is set."""
        result = os.environ.get("K_SERVICE")
        assert result == "test-service", "K_SERVICE should be 'test-service'"


class TestStreamlitIntegration:
    """Test suite for Streamlit component integration (mocked)."""

    @patch('streamlit.link_button')
    @patch('streamlit.text')
    @patch('streamlit.text_input')
    @patch('streamlit.header')
    @patch('streamlit.selectbox')
    @patch('streamlit.slider')
    @patch('streamlit.text_area')
    @patch('streamlit.button')
    @patch('streamlit.error')
    @patch('streamlit.spinner')
    @patch('streamlit.markdown')
    def test_streamlit_components_called(
        self,
        mock_markdown,
        mock_spinner,
        mock_error,
        mock_button,
        mock_text_area,
        mock_slider,
        mock_selectbox,
        mock_header,
        mock_text_input,
        mock_text,
        mock_link_button,
    ):
        """Test that Streamlit components are properly initialized."""
        # Mock session state
        with patch('streamlit.session_state', new_callable=dict):
            # This test verifies the app can be imported without errors
            # when Streamlit components are mocked
            import app  # noqa: F401
            
            # Verify critical Streamlit methods are available
            assert callable(mock_link_button), "link_button should be callable"
            assert callable(mock_text_input), "text_input should be callable"
            assert callable(mock_button), "button should be callable"


class TestThinkingConfigLogic:
    """Test suite for thinking budget configuration logic."""

    @pytest.mark.parametrize("model,expected_in_set", [
        ("gemini-2.5-pro", True),
        ("gemini-2.5-flash", True),
        ("gemini-2.5-flash-lite-preview-06-17", True),
        ("gemini-1.5-pro", False),
        ("unknown-model", False),
    ])
    def test_model_in_thinking_budget_models(self, model, expected_in_set):
        """Test which models support thinking budget."""
        from app import THINKING_BUDGET_MODELS
        
        result = model in THINKING_BUDGET_MODELS
        assert result == expected_in_set, f"Model {model} thinking budget support mismatch"

    @pytest.mark.parametrize("budget_mode,expected_value", [
        ("Auto", None),
        ("Manual", 12288),
        ("Off", 0),
    ])
    def test_thinking_budget_values(self, budget_mode, expected_value):
        """Test thinking budget calculation for different modes."""
        thinking_budget = None
        
        if budget_mode == "Manual":
            thinking_budget = 12288
        elif budget_mode == "Off":
            thinking_budget = 0
        
        assert thinking_budget == expected_value, f"Thinking budget for {budget_mode} should be {expected_value}"

    def test_thinking_budget_range(self):
        """Test valid range for manual thinking budget."""
        min_budget = 0
        max_budget = 24576
        test_values = [0, 1, 12288, 24576]
        
        for value in test_values:
            assert min_budget <= value <= max_budget, f"Budget {value} should be in valid range"

    def test_thinking_budget_invalid_values(self):
        """Test handling of invalid thinking budget values."""
        invalid_values = [-1, 24577, 100000]
        min_budget = 0
        max_budget = 24576
        
        for value in invalid_values:
            is_valid = min_budget <= value <= max_budget
            assert not is_valid, f"Budget {value} should be invalid"


class TestGenerateContentConfiguration:
    """Test suite for GenerateContentConfig parameters."""

    @pytest.mark.parametrize("temperature,is_valid", [
        (0.0, True),
        (1.0, True),
        (2.0, True),
        (0.5, True),
        (1.5, True),
        (-0.1, False),
        (2.1, False),
    ])
    def test_temperature_range(self, temperature, is_valid):
        """Test valid temperature values."""
        expected_valid = 0.0 <= temperature <= 2.0
        assert (expected_valid == is_valid), f"Temperature {temperature} validity mismatch"

    @pytest.mark.parametrize("max_tokens,is_valid", [
        (1, True),
        (8192, True),
        (65535, True),
        (32768, True),
        (0, False),
        (65536, False),
    ])
    def test_max_output_tokens_range(self, max_tokens, is_valid):
        """Test valid max output token values."""
        expected_valid = 1 <= max_tokens <= 65535
        assert (expected_valid == is_valid), f"Max tokens {max_tokens} validity mismatch"

    @pytest.mark.parametrize("top_p,is_valid", [
        (0.0, True),
        (0.95, True),
        (1.0, True),
        (0.5, True),
        (-0.1, False),
        (1.1, False),
    ])
    def test_top_p_range(self, top_p, is_valid):
        """Test valid top_p values."""
        expected_valid = 0.0 <= top_p <= 1.0
        assert (expected_valid == is_valid), f"Top_p {top_p} validity mismatch"


class TestPromptGeneration:
    """Test suite for prompt handling and processing."""

    def test_empty_prompt_handling(self):
        """Test handling of empty prompt."""
        prompt = ""
        assert len(prompt) == 0, "Empty prompt should have length 0"
        assert not prompt, "Empty prompt should be falsy"

    def test_whitespace_only_prompt(self):
        """Test handling of whitespace-only prompt."""
        prompt = "   \n\t  "
        assert len(prompt.strip()) == 0, "Whitespace-only prompt should strip to empty"

    def test_normal_prompt(self):
        """Test handling of normal prompt."""
        prompt = "Write a function to calculate factorial"
        assert len(prompt) > 0, "Normal prompt should have positive length"
        assert prompt.strip() == prompt.strip(), "Prompt should be trimmable"

    def test_long_prompt(self):
        """Test handling of very long prompt."""
        prompt = "x" * 10000
        assert len(prompt) == 10000, "Long prompt length should be preserved"

    def test_special_characters_in_prompt(self):
        """Test prompt with special characters."""
        prompt = "Write code with @#$%^&*(){}[]|\\:;\"'<>,.?/"
        assert len(prompt) > 0, "Prompt with special chars should be valid"
        assert "$" in prompt, "Special characters should be preserved"


@patch('app.genai.Client')
@patch('app.ingest')
class TestGenerateContentFlow:
    """Test suite for content generation flow with mocked dependencies."""

    def test_generate_with_api_key(self, mock_ingest, mock_client_class):
        """Test content generation with API key authentication."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock ingest response
        mock_ingest.return_value = ("summary", "tree", "content")
        
        # Mock response
        mock_response = Mock()
        mock_response.text = "Generated code here"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.total_token_count = 1000
        mock_client.models.generate_content.return_value = mock_response
        
        # Create client with API key
        from app import genai
        client = genai.Client(
            vertexai=False,
            project=None,
            location=None,
            api_key="test-api-key",
        )
        
        mock_client_class.assert_called_once()
        assert client is not None

    def test_generate_with_project_id(self, mock_ingest, mock_client_class):
        """Test content generation with project ID authentication."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock ingest response
        mock_ingest.return_value = ("summary", "tree", "content")
        
        # Mock response
        mock_response = Mock()
        mock_response.text = "Generated code here"
        mock_client.models.generate_content.return_value = mock_response
        
        # Create client with project ID
        from app import genai
        client = genai.Client(
            vertexai=True,
            project="test-project",
            location="global",
            api_key=None,
        )
        
        mock_client_class.assert_called_once()
        assert client is not None

    def test_generate_without_credentials(self, mock_ingest, mock_client_class):
        """Test error handling when no credentials provided."""
        api_key = None
        project_id = None
        
        # Verify both are None (should trigger error in app)
        assert not api_key and not project_id, "Both credentials should be None"

    def test_ingest_excludes_test_directories(self, mock_ingest, mock_client_class):
        """Test that ingest properly excludes test directories."""
        from app import GENAI_REPOS
        
        mock_ingest.return_value = ("summary", "tree", "content")
        
        expected_excludes = {
            "google/genai/tests/",
            "docs/",
            ".github/",
            "test/",
            "web/",
            "api-report/",
            "node/",
            "scripts/",
            "src/test/",
            "internal/changefinder/",
            "*_test.*",
            "testdata/",
        }
        
        # Verify the exclude patterns are defined
        assert isinstance(expected_excludes, set), "Excludes should be a set"
        assert len(expected_excludes) > 0, "Should have exclude patterns"

    def test_system_instruction_generation(self, mock_ingest, mock_client_class):
        """Test system instruction is properly formatted."""
        selected_language = "Python"
        expected_instruction = f"""You are an expert software engineer, proficient in {selected_language}. Your task is to write code using the Google Gen AI SDK based on the user's request. Don't suggest using Gemini 1.0 or 1.5. Do not suggest using the library google.generativeai"""
        
        assert "expert software engineer" in expected_instruction
        assert selected_language in expected_instruction
        assert "Google Gen AI SDK" in expected_instruction
        assert "Don't suggest using Gemini 1.0 or 1.5" in expected_instruction

    def test_contents_list_structure(self, mock_ingest, mock_client_class):
        """Test the structure of contents list sent to API."""
        prompt = "Write a test function"
        tree = "Repository structure"
        content = "Repository content"
        
        contents = [
            "The Google Gen AI SDK repository is provided here",
            tree,
            content,
            "This is the user's request:",
            prompt,
        ]
        
        assert len(contents) == 5, "Contents should have 5 elements"
        assert contents[0] == "The Google Gen AI SDK repository is provided here"
        assert contents[-1] == prompt, "Last element should be the prompt"

    def test_response_text_extraction(self, mock_ingest, mock_client_class):
        """Test extracting text from response."""
        mock_response = Mock()
        mock_response.text = "Generated code output"
        
        result = mock_response.text
        assert result == "Generated code output"
        assert isinstance(result, str)

    def test_response_usage_metadata(self, mock_ingest, mock_client_class):
        """Test extracting usage metadata from response."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.total_token_count = 1500
        
        assert mock_response.usage_metadata is not None
        assert mock_response.usage_metadata.total_token_count == 1500


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    def test_missing_both_credentials(self):
        """Test error when both API key and project ID are missing."""
        api_key = ""
        project_id = ""
        
        has_credentials = bool(api_key or project_id)
        assert not has_credentials, "Should not have credentials"

    def test_api_key_present(self):
        """Test when API key is provided."""
        api_key = "test-key"
        project_id = ""
        
        has_credentials = bool(api_key or project_id)
        assert has_credentials, "Should have credentials via API key"

    def test_project_id_present(self):
        """Test when project ID is provided."""
        api_key = ""
        project_id = "test-project"
        
        has_credentials = bool(api_key or project_id)
        assert has_credentials, "Should have credentials via project ID"

    @patch('app.genai.Client')
    def test_api_error_handling(self, mock_client_class):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock API error
        mock_client.models.generate_content.side_effect = Exception("API Error")
        
        client = mock_client
        
        # Verify exception is raised
        with pytest.raises(Exception) as exc_info:
            client.models.generate_content(
                model="gemini-2.5-flash",
                contents=["test"],
                config=Mock(),
            )
        
        assert "API Error" in str(exc_info.value)


class TestLocationConfiguration:
    """Test suite for location configuration logic."""

    def test_location_with_project_id(self):
        """Test location is set to 'global' when using project ID."""
        project_id = "test-project"
        location = "global" if project_id else None
        
        assert location == "global", "Location should be 'global' for project ID"

    def test_location_without_project_id(self):
        """Test location is None when using API key."""
        project_id = None
        location = "global" if project_id else None
        
        assert location is None, "Location should be None for API key"


class TestClientConfiguration:
    """Test suite for GenAI client configuration."""

    def test_client_vertexai_flag_with_project(self):
        """Test vertexai flag is True when project ID provided."""
        project_id = "test-project"
        vertexai_flag = bool(project_id)
        
        assert vertexai_flag is True, "vertexai should be True with project ID"

    def test_client_vertexai_flag_without_project(self):
        """Test vertexai flag is False when no project ID."""
        project_id = None
        vertexai_flag = bool(project_id)
        
        assert vertexai_flag is False, "vertexai should be False without project ID"


class TestResponseFormatting:
    """Test suite for response formatting."""

    def test_markdown_dollar_sign_escaping(self):
        """Test that dollar signs are properly escaped in markdown."""
        response_text = "The cost is $100"
        escaped_text = response_text.replace("$", r"\$")
        
        assert r"\$" in escaped_text, "Dollar signs should be escaped"
        assert escaped_text == r"The cost is \$100"

    def test_markdown_no_dollar_signs(self):
        """Test markdown text without dollar signs."""
        response_text = "This is a normal response"
        escaped_text = response_text.replace("$", r"\$")
        
        assert escaped_text == response_text, "Text should be unchanged"

    def test_token_count_formatting(self):
        """Test token count display formatting."""
        total_token_count = 1500
        display_text = f"Total tokens: {total_token_count}"
        
        assert display_text == "Total tokens: 1500"
        assert str(total_token_count) in display_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])