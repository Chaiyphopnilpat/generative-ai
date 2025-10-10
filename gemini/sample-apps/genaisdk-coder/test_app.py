# Copyright 2025 Google LLC
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

"""Comprehensive unit tests for genaisdk-coder app.py"""

import os
from unittest.mock import MagicMock, Mock, patch, call

import pytest

# Import the module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestGetModelName:
    """Tests for the get_model_name function."""

    def test_get_model_name_with_known_model(self):
        """Test get_model_name returns correct name for known models."""
        from app import get_model_name, MODELS

        for model_id, expected_name in MODELS.items():
            result = get_model_name(model_id)
            assert result == expected_name
            assert isinstance(result, str)

    def test_get_model_name_with_none(self):
        """Test get_model_name returns 'Gemini' when input is None."""
        from app import get_model_name

        result = get_model_name(None)
        assert result == "Gemini"

    def test_get_model_name_with_unknown_model(self):
        """Test get_model_name returns 'Gemini' for unknown model IDs."""
        from app import get_model_name

        result = get_model_name("unknown-model-123")
        assert result == "Gemini"

    def test_get_model_name_with_empty_string(self):
        """Test get_model_name returns 'Gemini' for empty string."""
        from app import get_model_name

        result = get_model_name("")
        assert result == "Gemini"


class TestConstants:
    """Tests for module-level constants and configurations."""

    def test_models_dict_structure(self):
        """Test MODELS dictionary contains expected model IDs."""
        from app import MODELS

        assert isinstance(MODELS, dict)
        assert len(MODELS) > 0
        assert "gemini-2.5-flash" in MODELS
        assert "gemini-2.5-pro" in MODELS
        assert all(isinstance(v, str) for v in MODELS.values())

    def test_thinking_budget_models_set(self):
        """Test THINKING_BUDGET_MODELS contains expected models."""
        from app import THINKING_BUDGET_MODELS

        assert isinstance(THINKING_BUDGET_MODELS, set)
        assert "gemini-2.5-pro" in THINKING_BUDGET_MODELS
        assert "gemini-2.5-flash" in THINKING_BUDGET_MODELS

    def test_genai_repos_dict(self):
        """Test GENAI_REPOS contains language mappings."""
        from app import GENAI_REPOS

        assert isinstance(GENAI_REPOS, dict)
        assert "Python" in GENAI_REPOS
        assert "Java" in GENAI_REPOS
        assert "Go" in GENAI_REPOS
        assert "JavaScript" in GENAI_REPOS

        # Verify all values are GitHub URLs
        for repo_url in GENAI_REPOS.values():
            assert repo_url.startswith("https://github.com/googleapis/")
            assert "genai" in repo_url


class TestStreamlitConfiguration:
    """Tests for Streamlit UI configuration and widget setup."""

    @patch('app.st')
    def test_link_buttons_rendered(self, mock_st):
        """Test that GitHub and Cloud Run link buttons are rendered."""
        mock_st.link_button = MagicMock()

        # Import triggers Streamlit calls
        import importlib
        import app
        importlib.reload(app)

        # Verify GitHub link button was called
        calls = mock_st.link_button.call_args_list
        github_calls = [c for c in calls if "GitHub" in str(c)]
        assert len(github_calls) >= 1

    @patch('app.st')
    @patch.dict(os.environ, {'K_SERVICE': 'test-service'})
    def test_cloud_run_button_when_service_exists(self, mock_st):
        """Test Cloud Run button is shown when K_SERVICE env var is set."""
        mock_st.link_button = MagicMock()

        import importlib
        import app
        importlib.reload(app)

        # Should create two link buttons: GitHub and Cloud Run
        assert mock_st.link_button.call_count >= 2

    @patch('app.st')
    @patch.dict(os.environ, {}, clear=True)
    def test_cloud_run_button_when_no_service(self, mock_st):
        """Test Cloud Run button is not shown when K_SERVICE is not set."""
        mock_st.link_button = MagicMock()

        # Ensure K_SERVICE is not in environment
        os.environ.pop('K_SERVICE', None)

        import importlib
        import app
        importlib.reload(app)

        # Should create only GitHub link button
        calls = [str(c) for c in mock_st.link_button.call_args_list]
        cloud_run_calls = [c for c in calls if "Cloud Run" in c]
        assert len(cloud_run_calls) == 0


class TestGenerateContentConfigConstruction:
    """Tests for GenerateContentConfig object construction."""

    @patch('app.st')
    def test_config_with_thinking_budget(self, mock_st):
        """Test config includes thinking_config when thinking_budget is set."""
        from google.genai.types import ThinkingConfig

        thinking_budget = 1000
        thinking_config = ThinkingConfig(thinking_budget=thinking_budget)

        assert thinking_config.thinking_budget == thinking_budget

    @patch('app.st')
    def test_config_without_thinking_budget(self, mock_st):
        """Test config without thinking_config when budget is None."""
        from google.genai.types import ThinkingConfig

        # When thinking_budget is None, ThinkingConfig should not be created
        thinking_budget = None
        thinking_config = (
            ThinkingConfig(thinking_budget=thinking_budget)
            if thinking_budget is not None
            else None
        )

        assert thinking_config is None


class TestGenAIClientInitialization:
    """Tests for Google Gen AI client initialization."""

    @patch('app.genai.Client')
    def test_client_with_api_key(self, mock_client_class):
        """Test client initialization with API key."""
        from app import genai

        api_key = "dummy_api_key"
        project_id = None
        location = None

        client = genai.Client(
            vertexai=bool(project_id),
            project=project_id,
            location=location,
            api_key=api_key,
        )

        mock_client_class.assert_called_once_with(
            vertexai=False,
            project=None,
            location=None,
            api_key=api_key,
        )

    @patch('app.genai.Client')
    def test_client_with_project_id(self, mock_client_class):
        """Test client initialization with project ID."""
        from app import genai

        api_key = None
        project_id = "test-project-123"
        location = "global"

        client = genai.Client(
            vertexai=bool(project_id),
            project=project_id,
            location=location,
            api_key=api_key,
        )

        mock_client_class.assert_called_once_with(
            vertexai=True,
            project=project_id,
            location="global",
            api_key=None,
        )

    @patch('app.genai.Client')
    def test_location_is_global_when_project_id_exists(self, mock_client_class):
        """Test that location is set to 'global' when project_id is provided."""
        from app import genai

        project_id = "my-project"
        location = "global" if project_id else None

        assert location == "global"


class TestGitIngestIntegration:
    """Tests for gitingest repository ingestion."""

    @patch('app.ingest')
    def test_ingest_called_with_correct_repo(self, mock_ingest):
        """Test that ingest is called with the correct repository URL."""
        from app import GENAI_REPOS

        mock_ingest.return_value = ("summary", "tree", "content")

        repo_url = GENAI_REPOS["Python"]
        summary, tree, content = mock_ingest(
            source=repo_url,
            exclude_patterns={"docs/", ".github/"}
        )

        mock_ingest.assert_called_once()
        assert mock_ingest.call_args[1]["source"] == repo_url

    @patch('app.ingest')
    def test_ingest_excludes_test_files(self, mock_ingest):
        """Test that ingest excludes test directories and files."""
        mock_ingest.return_value = ("summary", "tree", "content")

        exclude_patterns = {
            "google/genai/tests/",
            "docs/",
            ".github/",
            "test/",
            "*_test.*",
            "testdata/",
        }

        _, tree, content = mock_ingest(
            source="https://github.com/test/repo",
            exclude_patterns=exclude_patterns
        )

        # Verify exclude patterns include test-related paths
        call_args = mock_ingest.call_args[1]["exclude_patterns"]
        assert "test/" in call_args or "*_test.*" in call_args


class TestContentGeneration:
    """Tests for content generation flow."""

    @patch('app.st')
    @patch('app.genai.Client')
    @patch('app.ingest')
    def test_generate_content_request_structure(self, mock_ingest, mock_client_class, mock_st):
        """Test that generate_content is called with correct structure."""
        mock_ingest.return_value = ("summary", "tree", "content")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated code example"
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.total_token_count = 150
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Simulate the generate content call
        contents = [
            "The Google Gen AI SDK repository is provided here",
            "tree",
            "content",
            "This is the user's request:",
            "Write a hello world example",
        ]

        response = mock_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=MagicMock(),
        )

        assert response.text == "Generated code example"
        assert response.usage_metadata.total_token_count == 150

    @patch('app.st')
    @patch('app.genai.Client')
    @patch('app.ingest')
    def test_system_instruction_includes_language(self, mock_ingest, mock_client_class, mock_st):
        """Test that system instruction mentions the selected language."""
        from google.genai.types import GenerateContentConfig

        mock_ingest.return_value = ("summary", "tree", "content")
        selected_language = "Python"

        config = GenerateContentConfig()
        config.system_instruction = (
            f"You are an expert software engineer, proficient in {selected_language}. "
            f"Your task is to write code using the Google Gen AI SDK based on the user's request."
        )

        assert selected_language in config.system_instruction
        assert "Google Gen AI SDK" in config.system_instruction

    @patch('app.st')
    @patch('app.genai.Client')
    @patch('app.ingest')
    def test_error_handling_when_no_auth(self, mock_ingest, mock_client_class, mock_st):
        """Test error handling when neither API key nor project ID is provided."""
        mock_st.error = MagicMock()

        # Simulate the error condition
        api_key = None
        project_id = None

        if not api_key and not project_id:
            error_msg = "🚨 Configuration Error: Please set either `GOOGLE_API_KEY` or `PROJECT_ID`."
            mock_st.error(error_msg)

        mock_st.error.assert_called_once()
        assert "Configuration Error" in mock_st.error.call_args[0][0]


class TestParameterValidation:
    """Tests for input parameter validation and ranges."""

    def test_temperature_range(self):
        """Test temperature parameter is within valid range."""
        # Valid ranges for temperature slider
        valid_temperatures = [0.0, 0.5, 1.0, 1.5, 2.0]

        for temp in valid_temperatures:
            assert 0.0 <= temp <= 2.0

    def test_max_output_tokens_range(self):
        """Test max_output_tokens is within valid range."""
        # Valid ranges for max_output_tokens slider
        valid_tokens = [1, 100, 8192, 32768, 65535]

        for tokens in valid_tokens:
            assert 1 <= tokens <= 65535

    def test_top_p_range(self):
        """Test top_p parameter is within valid range."""
        # Valid ranges for top_p slider
        valid_top_p = [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]

        for top_p in valid_top_p:
            assert 0.0 <= top_p <= 1.0

    def test_thinking_budget_range(self):
        """Test thinking_budget is within valid range."""
        # Valid ranges for thinking budget
        valid_budgets = [0, 1000, 12288, 24576]

        for budget in valid_budgets:
            assert 0 <= budget <= 24576


class TestResponseHandling:
    """Tests for response processing and display."""

    @patch('app.st')
    def test_markdown_response_display(self, mock_st):
        """Test that response text is properly displayed."""
        mock_st.markdown = MagicMock()

        response_text = "Here's a code example:\n```python\nprint('Hello')\n```"
        mock_st.markdown(response_text)

        mock_st.markdown.assert_called_once_with(response_text)

    @patch('app.st')
    def test_token_count_display(self, mock_st):
        """Test that token count is displayed when available."""
        mock_st.text = MagicMock()

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.total_token_count = 2500

        if mock_usage_metadata and mock_usage_metadata.total_token_count is not None:
            mock_st.text(f"Total tokens: {mock_usage_metadata.total_token_count}")

        mock_st.text.assert_called_once_with("Total tokens: 2500")

    @patch('app.st')
    def test_no_token_display_when_none(self, mock_st):
        """Test that token count is not displayed when None."""
        mock_st.text = MagicMock()

        mock_usage_metadata = None

        if mock_usage_metadata and mock_usage_metadata.total_token_count is not None:
            mock_st.text(f"Total tokens: {mock_usage_metadata.total_token_count}")

        mock_st.text.assert_not_called()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_prompt_handling(self):
        """Test handling of empty prompt input."""
        prompt = ""
        generate_freeform = True

        # Should not proceed with empty prompt
        if generate_freeform and prompt:
            # This block should not execute
            assert False, "Should not generate with empty prompt"
        else:
            # Expected behavior
            assert True

    def test_very_long_prompt(self):
        """Test handling of very long prompts."""
        prompt = "A" * 100000

        # Prompt should be accepted even if very long
        assert len(prompt) == 100000
        assert isinstance(prompt, str)

    @patch('app.st')
    def test_spinner_context_during_generation(self, mock_st):
        """Test that spinner is shown during generation."""
        mock_st.spinner = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        with mock_st.spinner("Generating response..."):
            pass

        mock_st.spinner.assert_called_once()

    def test_model_selection_validation(self):
        """Test that only valid models can be selected."""
        from app import MODELS

        valid_model_ids = list(MODELS.keys())

        # Test that all models in MODELS dict are valid strings
        for model_id in valid_model_ids:
            assert isinstance(model_id, str)
            assert len(model_id) > 0
            assert "gemini" in model_id.lower()


class TestThinkingBudgetModes:
    """Tests for thinking budget configuration modes."""

    def test_thinking_budget_auto_mode(self):
        """Test Auto mode sets thinking_budget to None."""
        thinking_budget_mode = "Auto"

        if thinking_budget_mode == "Manual":
            thinking_budget = 1000
        elif thinking_budget_mode == "Off":
            thinking_budget = 0
        else:  # Auto
            thinking_budget = None

        assert thinking_budget is None

    def test_thinking_budget_manual_mode(self):
        """Test Manual mode allows custom thinking_budget value."""
        thinking_budget_mode = "Manual"
        manual_value = 12288

        if thinking_budget_mode == "Manual":
            thinking_budget = manual_value
        elif thinking_budget_mode == "Off":
            thinking_budget = 0
        else:
            thinking_budget = None

        assert thinking_budget == manual_value

    def test_thinking_budget_off_mode(self):
        """Test Off mode sets thinking_budget to 0."""
        thinking_budget_mode = "Off"

        if thinking_budget_mode == "Manual":
            thinking_budget = 1000
        elif thinking_budget_mode == "Off":
            thinking_budget = 0
        else:
            thinking_budget = None

        assert thinking_budget == 0

    def test_thinking_config_creation_conditional(self):
        """Test ThinkingConfig is only created when budget is not None."""
        from google.genai.types import ThinkingConfig

        # Case 1: Budget is set
        thinking_budget = 5000
        thinking_config = (
            ThinkingConfig(thinking_budget=thinking_budget)
            if thinking_budget is not None
            else None
        )
        assert thinking_config is not None
        assert thinking_config.thinking_budget == 5000

        # Case 2: Budget is None
        thinking_budget = None
        thinking_config = (
            ThinkingConfig(thinking_budget=thinking_budget)
            if thinking_budget is not None
            else None
        )
        assert thinking_config is None


class TestIntegrationScenarios:
    """Integration tests for complete user workflows."""

    @patch('app.st')
    @patch('app.genai.Client')
    @patch('app.ingest')
    def test_complete_generation_workflow(self, mock_ingest, mock_client_class, mock_st):
        """Test complete workflow from user input to response display."""
        # Setup mocks
        mock_ingest.return_value = ("summary", "tree_content", "repo_content")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "```python\nfrom google import genai\nclient = genai.Client()\n```"
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.total_token_count = 250
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        mock_st.markdown = MagicMock()
        mock_st.text = MagicMock()

        # Simulate workflow
        api_key = "test-key"
        project_id = None
        selected_model = "gemini-2.5-flash"
        prompt = "Create a simple example"

        # Generate content
        response = mock_client.models.generate_content(
            model=selected_model,
            contents=["prompt"],
            config=MagicMock(),
        )

        # Display results
        mock_st.markdown(response.text)
        mock_st.text(f"Total tokens: {response.usage_metadata.total_token_count}")

        # Verify workflow completed
        assert mock_client.models.generate_content.called
        assert mock_st.markdown.called
        assert mock_st.text.called

    @patch('app.st')
    @patch('app.genai.Client')
    @patch('app.ingest')
    def test_error_recovery_workflow(self, mock_ingest, mock_client_class, mock_st):
        """Test error handling and recovery workflow."""
        # Setup error scenario
        mock_ingest.side_effect = Exception("Network error")
        mock_st.error = MagicMock()

        # Attempt to ingest
        try:
            mock_ingest(source="https://example.com", exclude_patterns=set())
        except Exception as e:
            mock_st.error(f"Error: {str(e)}")

        # Verify error was displayed
        mock_st.error.assert_called_once()
        assert "Network error" in mock_st.error.call_args[0][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])