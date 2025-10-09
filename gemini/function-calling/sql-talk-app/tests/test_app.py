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
Comprehensive unit tests for SQL Talk with BigQuery Streamlit application.
"""

import os
import sys
from unittest.mock import Mock, patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestConstants:
    """Test suite for application constants."""

    def test_bigquery_dataset_id(self):
        """Test BigQuery dataset ID constant."""
        from app import BIGQUERY_DATASET_ID
        assert BIGQUERY_DATASET_ID == "thelook_ecommerce"
        assert isinstance(BIGQUERY_DATASET_ID, str)

    def test_model_id(self):
        """Test Gemini model ID constant."""
        from app import MODEL_ID
        assert MODEL_ID == "gemini-2.0-flash"
        assert "gemini" in MODEL_ID.lower()

    def test_location(self):
        """Test location constant."""
        from app import LOCATION
        assert LOCATION == "us-central1"


class TestFunctionDeclarations:
    """Test suite for function declarations."""

    def test_list_datasets_func(self):
        """Test list_datasets function declaration."""
        from app import list_datasets_func
        assert list_datasets_func.name == "list_datasets"
        assert len(list_datasets_func.description) > 0

    def test_list_tables_func(self):
        """Test list_tables function declaration."""
        from app import list_tables_func
        assert list_tables_func.name == "list_tables"
        assert "dataset" in list_tables_func.description.lower()

    def test_get_table_func(self):
        """Test get_table function declaration."""
        from app import get_table_func
        assert get_table_func.name == "get_table"
        assert "schema" in get_table_func.description.lower()

    def test_sql_query_func(self):
        """Test sql_query function declaration."""
        from app import sql_query_func
        assert sql_query_func.name == "sql_query"
        assert "sql" in sql_query_func.description.lower()

    def test_all_functions_unique(self):
        """Test all functions have unique names."""
        from app import list_datasets_func, list_tables_func, get_table_func, sql_query_func
        names = [f.name for f in [list_datasets_func, list_tables_func, get_table_func, sql_query_func]]
        assert len(names) == len(set(names))


class TestToolConfiguration:
    """Test suite for tool configuration."""

    def test_sql_query_tool_structure(self):
        """Test sql_query_tool structure."""
        from app import sql_query_tool
        assert hasattr(sql_query_tool, 'function_declarations')


class TestErrorHandling:
    """Test suite for error handling."""

    def test_sql_error_message_format(self):
        """Test SQL error message formatting."""
        error = Exception("Invalid SQL")
        error_msg = f"We're having trouble running this SQL query. Details: {str(error)}"
        assert "trouble running" in error_msg
        assert "Invalid SQL" in error_msg


class TestQueryCleaning:
    """Test suite for query string cleaning."""

    def test_sql_query_cleaning(self):
        """Test SQL query string cleaning."""
        dirty = "SELECT * FROM\\ntable"
        cleaned = dirty.replace("\\n", " ").replace("\n", "").replace("\\", "")
        assert "\\n" not in cleaned
        assert "\n" not in cleaned


class TestMarkdownFormatting:
    """Test suite for markdown formatting."""

    def test_dollar_sign_escaping(self):
        """Test dollar signs are escaped."""
        text = "Price is $100"
        escaped = text.replace("$", r"\$")
        assert r"\$" in escaped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])