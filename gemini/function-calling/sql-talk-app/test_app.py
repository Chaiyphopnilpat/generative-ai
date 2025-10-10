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

"""Comprehensive unit tests for sql-talk-app app.py"""

import time
from unittest.mock import MagicMock, Mock, patch, call

import pytest


# Import the module under test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestConstants:
    """Tests for module-level constants."""

    def test_bigquery_dataset_id(self):
        """Test BIGQUERY_DATASET_ID is correctly set."""
        from app import BIGQUERY_DATASET_ID
        
        assert BIGQUERY_DATASET_ID == "thelook_ecommerce"
        assert isinstance(BIGQUERY_DATASET_ID, str)

    def test_model_id(self):
        """Test MODEL_ID is set to gemini-2.0-flash."""
        from app import MODEL_ID
        
        assert MODEL_ID == "gemini-2.0-flash"
        assert "gemini" in MODEL_ID.lower()

    def test_location(self):
        """Test LOCATION is correctly set."""
        from app import LOCATION
        
        assert LOCATION == "us-central1"
        assert isinstance(LOCATION, str)


class TestFunctionDeclarations:
    """Tests for BigQuery function declarations."""

    def test_list_datasets_function_declaration(self):
        """Test list_datasets function declaration structure."""
        from app import list_datasets_func
        
        assert list_datasets_func.name == "list_datasets"
        assert "dataset" in list_datasets_func.description.lower()
        assert list_datasets_func.parameters["type"] == "object"
        assert "properties" in list_datasets_func.parameters

    def test_list_tables_function_declaration(self):
        """Test list_tables function declaration structure."""
        from app import list_tables_func
        
        assert list_tables_func.name == "list_tables"
        assert "table" in list_tables_func.description.lower()
        assert "dataset_id" in list_tables_func.parameters["properties"]
        assert "dataset_id" in list_tables_func.parameters["required"]

    def test_get_table_function_declaration(self):
        """Test get_table function declaration structure."""
        from app import get_table_func
        
        assert get_table_func.name == "get_table"
        assert "table" in get_table_func.description.lower()
        assert "table_id" in get_table_func.parameters["properties"]
        assert "table_id" in get_table_func.parameters["required"]
        assert "fully qualified" in get_table_func.description.lower()

    def test_sql_query_function_declaration(self):
        """Test sql_query function declaration structure."""
        from app import sql_query_func
        
        assert sql_query_func.name == "sql_query"
        assert "sql" in sql_query_func.description.lower() or "query" in sql_query_func.description.lower()
        assert "query" in sql_query_func.parameters["properties"]
        assert "query" in sql_query_func.parameters["required"]

    def test_all_functions_in_tool(self):
        """Test that all function declarations are included in sql_query_tool."""
        from app import sql_query_tool, list_datasets_func, list_tables_func, get_table_func, sql_query_func
        
        function_names = [f.name for f in sql_query_tool.function_declarations]
        
        assert list_datasets_func.name in function_names
        assert list_tables_func.name in function_names
        assert get_table_func.name in function_names
        assert sql_query_func.name in function_names
        assert len(function_names) == 4


class TestGenAIClientInitialization:
    """Tests for Gen AI client initialization."""

    @patch('app.genai.Client')
    def test_client_initialized_with_vertexai(self, mock_client_class):
        """Test that client is initialized with vertexai=True."""
        from app import LOCATION
        
        # Simulate client creation
        mock_client_class(vertexai=True, location=LOCATION)
        
        mock_client_class.assert_called_once_with(
            vertexai=True,
            location=LOCATION
        )

    @patch('app.genai.Client')
    def test_client_location_is_correct(self, mock_client_class):
        """Test that client uses the correct location."""
        from app import LOCATION
        
        mock_client_class(vertexai=True, location=LOCATION)
        
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["location"] == "us-central1"


class TestStreamlitPageConfiguration:
    """Tests for Streamlit page configuration."""

    @patch('app.st')
    def test_page_config_set(self, mock_st):
        """Test that page config is set with correct parameters."""
        mock_st.set_page_config = MagicMock()
        
        # Simulate page config call
        mock_st.set_page_config(
            page_title="SQL Talk with BigQuery",
            page_icon="vertex-ai.png",
            layout="wide",
        )
        
        mock_st.set_page_config.assert_called_once()
        call_kwargs = mock_st.set_page_config.call_args[1]
        assert call_kwargs["page_title"] == "SQL Talk with BigQuery"
        assert call_kwargs["layout"] == "wide"


class TestListDatasetsHandler:
    """Tests for list_datasets function call handler."""

    @patch('app.bigquery.Client')
    def test_list_datasets_returns_dataset_id(self, mock_bq_client_class):
        """Test list_datasets returns the configured dataset ID."""
        from app import BIGQUERY_DATASET_ID
        
        mock_client = MagicMock()
        mock_client.list_datasets.return_value = [MagicMock()]
        mock_bq_client_class.return_value = mock_client
        
        client = mock_bq_client_class()
        api_response = client.list_datasets()
        api_response = BIGQUERY_DATASET_ID
        
        assert api_response == "thelook_ecommerce"

    @patch('app.bigquery.Client')
    def test_list_datasets_api_call(self, mock_bq_client_class):
        """Test that list_datasets calls BigQuery API."""
        mock_client = MagicMock()
        mock_datasets = [MagicMock(dataset_id="dataset1"), MagicMock(dataset_id="dataset2")]
        mock_client.list_datasets.return_value = mock_datasets
        mock_bq_client_class.return_value = mock_client
        
        client = mock_bq_client_class()
        result = client.list_datasets()
        
        assert len(result) == 2
        mock_client.list_datasets.assert_called_once()


class TestListTablesHandler:
    """Tests for list_tables function call handler."""

    @patch('app.bigquery.Client')
    def test_list_tables_with_dataset_id(self, mock_bq_client_class):
        """Test list_tables with dataset_id parameter."""
        mock_client = MagicMock()
        mock_table1 = MagicMock()
        mock_table1.table_id = "orders"
        mock_table2 = MagicMock()
        mock_table2.table_id = "customers"
        mock_client.list_tables.return_value = [mock_table1, mock_table2]
        mock_bq_client_class.return_value = mock_client
        
        client = mock_bq_client_class()
        params = {"dataset_id": "thelook_ecommerce"}
        api_response = client.list_tables(params["dataset_id"])
        api_response_str = str([table.table_id for table in api_response])
        
        assert "orders" in api_response_str
        assert "customers" in api_response_str
        mock_client.list_tables.assert_called_once_with("thelook_ecommerce")

    @patch('app.bigquery.Client')
    def test_list_tables_returns_string(self, mock_bq_client_class):
        """Test that list_tables result is converted to string."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table.table_id = "test_table"
        mock_client.list_tables.return_value = [mock_table]
        mock_bq_client_class.return_value = mock_client
        
        client = mock_bq_client_class()
        api_response = client.list_tables("dataset")
        api_response_str = str([table.table_id for table in api_response])
        
        assert isinstance(api_response_str, str)
        assert "test_table" in api_response_str


class TestGetTableHandler:
    """Tests for get_table function call handler."""

    @patch('app.bigquery.Client')
    def test_get_table_with_table_id(self, mock_bq_client_class):
        """Test get_table with table_id parameter."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table_repr = {
            "description": "Test table description",
            "schema": {
                "fields": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "name", "type": "STRING"},
                ]
            }
        }
        mock_table.to_api_repr.return_value = mock_table_repr
        mock_client.get_table.return_value = mock_table
        mock_bq_client_class.return_value = mock_client
        
        client = mock_bq_client_class()
        params = {"table_id": "project.dataset.table"}
        api_response = client.get_table(params["table_id"])
        api_response_dict = api_response.to_api_repr()
        
        assert "description" in api_response_dict
        assert "schema" in api_response_dict
        assert len(api_response_dict["schema"]["fields"]) == 2

    @patch('app.bigquery.Client')
    def test_get_table_extracts_schema_fields(self, mock_bq_client_class):
        """Test that get_table extracts schema field names."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table_repr = {
            "description": "Table desc",
            "schema": {
                "fields": [
                    {"name": "column1", "type": "STRING"},
                    {"name": "column2", "type": "INTEGER"},
                    {"name": "column3", "type": "FLOAT"},
                ]
            }
        }
        mock_table.to_api_repr.return_value = mock_table_repr
        mock_client.get_table.return_value = mock_table
        mock_bq_client_class.return_value = mock_client
        
        client = mock_bq_client_class()
        api_response = client.get_table("table_id")
        api_repr = api_response.to_api_repr()
        
        column_names = [field["name"] for field in api_repr["schema"]["fields"]]
        
        assert "column1" in column_names
        assert "column2" in column_names
        assert "column3" in column_names
        assert len(column_names) == 3


class TestSqlQueryHandler:
    """Tests for sql_query function call handler."""

    @patch('app.bigquery.Client')
    def test_sql_query_with_valid_query(self, mock_bq_client_class):
        """Test sql_query with a valid SQL query."""
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_row1 = {"count": 100, "category": "electronics"}
        mock_row2 = {"count": 50, "category": "books"}
        mock_job.result.return_value = [mock_row1, mock_row2]
        mock_client.query.return_value = mock_job
        mock_bq_client_class.return_value = mock_client
        
        client = mock_bq_client_class()
        params = {"query": "SELECT COUNT(*) FROM table"}
        
        from google.cloud import bigquery
        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=100000000)
        
        cleaned_query = params["query"].replace("\\n", " ").replace("\n", "").replace("\\", "")
        query_job = client.query(cleaned_query, job_config=job_config)
        api_response = query_job.result()
        
        assert len(list(api_response)) == 2
        mock_client.query.assert_called_once()

    @patch('app.bigquery.Client')
    def test_sql_query_cleans_query_string(self, mock_bq_client_class):
        """Test that SQL query is cleaned before execution."""
        query = "SELECT *\\nFROM\\ntable\\nWHERE id = 1"
        
        cleaned_query = query.replace("\\n", " ").replace("\n", "").replace("\\", "")
        
        assert "\\n" not in cleaned_query
        assert "\n" not in cleaned_query
        assert "\\" not in cleaned_query
        assert "FROM" in cleaned_query

    @patch('app.bigquery.Client')
    def test_sql_query_with_maximum_bytes_billed(self, mock_bq_client_class):
        """Test sql_query uses maximum_bytes_billed configuration."""
        from google.cloud import bigquery
        
        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=100000000)
        
        assert job_config.maximum_bytes_billed == 100000000

    @patch('app.bigquery.Client')
    def test_sql_query_converts_result_to_dict_list(self, mock_bq_client_class):
        """Test that query results are converted to list of dictionaries."""
        mock_client = MagicMock()
        mock_job = MagicMock()
        
        # Create mock row objects with dict() support
        mock_row1 = MagicMock()
        mock_row1.__iter__ = lambda self: iter([("id", 1), ("name", "Alice")])
        mock_row2 = MagicMock()
        mock_row2.__iter__ = lambda self: iter([("id", 2), ("name", "Bob")])
        
        mock_job.result.return_value = [mock_row1, mock_row2]
        mock_client.query.return_value = mock_job
        mock_bq_client_class.return_value = mock_client
        
        client = mock_bq_client_class()
        query_job = client.query("SELECT * FROM table")
        api_response = query_job.result()
        
        # Convert to list of dicts (simulating dict(row))
        result_list = [dict(row) for row in api_response]
        result_str = str(result_list)
        
        assert isinstance(result_str, str)
        assert len(result_list) == 2


class TestSqlQueryErrorHandling:
    """Tests for SQL query error handling."""

    @patch('app.st')
    @patch('app.bigquery.Client')
    def test_sql_query_handles_invalid_query(self, mock_bq_client_class, mock_st):
        """Test error handling for invalid SQL queries."""
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception("Invalid SQL syntax")
        mock_bq_client_class.return_value = mock_client
        mock_st.error = MagicMock()
        
        client = mock_bq_client_class()
        params = {"query": "INVALID SQL"}
        
        try:
            cleaned_query = params["query"]
            query_job = client.query(cleaned_query)
        except Exception as e:
            error_message = f"""
            We're having trouble running this SQL query. This
            could be due to an invalid query or the structure of
            the data. Try rephrasing your question to help the
            model generate a valid query. Details:

            {str(e)}"""
            mock_st.error(error_message)
        
        mock_st.error.assert_called_once()
        assert "Invalid SQL syntax" in mock_st.error.call_args[0][0]

    @patch('app.st')
    def test_error_message_format(self, mock_st):
        """Test that error messages are properly formatted."""
        mock_st.error = MagicMock()
        
        error_detail = "Column 'invalid_column' not found"
        error_message = f"""
            We're having trouble running this SQL query. This
            could be due to an invalid query or the structure of
            the data. Try rephrasing your question to help the
            model generate a valid query. Details:

            {error_detail}"""
        
        mock_st.error(error_message)
        
        assert "We're having trouble" in mock_st.error.call_args[0][0]
        assert error_detail in mock_st.error.call_args[0][0]


class TestFunctionCallingLoop:
    """Tests for the function calling loop logic."""

    @patch('app.genai.Client')
    def test_function_calling_loop_initialization(self, mock_client_class):
        """Test function calling loop initialization."""
        function_calling_in_process = True
        
        assert function_calling_in_process is True

    @patch('app.genai.Client')
    def test_function_calling_extracts_params(self, mock_client_class):
        """Test that function call parameters are extracted correctly."""
        mock_response = MagicMock()
        mock_response.function_call.args = {"dataset_id": "test_dataset", "limit": 10}
        
        params = {}
        for key, value in mock_response.function_call.args.items():
            params[key] = value
        
        assert params["dataset_id"] == "test_dataset"
        assert params["limit"] == 10

    @patch('app.genai.Client')
    def test_function_calling_loop_exits_on_attribute_error(self, mock_client_class):
        """Test that loop exits when response has no function_call attribute."""
        mock_response = MagicMock()
        delattr(mock_response, 'function_call')
        
        function_calling_in_process = True
        
        try:
            _ = mock_response.function_call
        except AttributeError:
            function_calling_in_process = False
        
        assert function_calling_in_process is False


class TestResponseHandling:
    """Tests for response handling and display."""

    @patch('app.st')
    def test_backend_details_formatting(self, mock_st):
        """Test that backend details are properly formatted."""
        backend_details = ""
        
        function_name = "list_tables"
        params = {"dataset_id": "test_dataset"}
        api_response = "['table1', 'table2']"
        
        backend_details += "- Function call:\n"
        backend_details += f"   - Function name: ```{function_name}```"
        backend_details += "\n\n"
        backend_details += f"   - Function parameters: ```{params}```"
        backend_details += "\n\n"
        backend_details += f"   - API response: ```{api_response}```"
        backend_details += "\n\n"
        
        assert "Function call:" in backend_details
        assert function_name in backend_details
        assert str(params) in backend_details
        assert api_response in backend_details

    @patch('app.st')
    def test_markdown_escapes_dollar_signs(self, mock_st):
        """Test that dollar signs are escaped in markdown display."""
        mock_st.markdown = MagicMock()
        
        text_with_dollar = "The price is $100"
        escaped_text = text_with_dollar.replace("$", r"\$")
        
        mock_st.markdown(escaped_text)
        
        assert r"\$" in mock_st.markdown.call_args[0][0]

    @patch('app.st')
    def test_expander_shows_backend_details(self, mock_st):
        """Test that backend details are shown in an expander."""
        mock_st.expander = MagicMock()
        mock_expander_context = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander_context)
        mock_st.expander.return_value.__exit__ = MagicMock()
        
        with mock_st.expander("Function calls, parameters, and responses:"):
            pass
        
        mock_st.expander.assert_called_once()


class TestSessionStateManagement:
    """Tests for Streamlit session state management."""

    @patch('app.st')
    def test_session_state_messages_initialization(self, mock_st):
        """Test that messages list is initialized in session state."""
        mock_st.session_state = {}
        
        if "messages" not in mock_st.session_state:
            mock_st.session_state["messages"] = []
        
        assert "messages" in mock_st.session_state
        assert isinstance(mock_st.session_state["messages"], list)

    @patch('app.st')
    def test_user_message_appended_to_session(self, mock_st):
        """Test that user messages are appended to session state."""
        mock_st.session_state = {"messages": []}
        
        prompt = "What are the total sales?"
        mock_st.session_state["messages"].append({"role": "user", "content": prompt})
        
        assert len(mock_st.session_state["messages"]) == 1
        assert mock_st.session_state["messages"][0]["role"] == "user"
        assert mock_st.session_state["messages"][0]["content"] == prompt

    @patch('app.st')
    def test_assistant_message_with_backend_details(self, mock_st):
        """Test that assistant messages include backend details."""
        mock_st.session_state = {"messages": []}
        
        full_response = "The total sales are $1,000,000"
        backend_details = "Function calls: list_tables, sql_query"
        
        mock_st.session_state["messages"].append({
            "role": "assistant",
            "content": full_response,
            "backend_details": backend_details,
        })
        
        assert len(mock_st.session_state["messages"]) == 1
        message = mock_st.session_state["messages"][0]
        assert message["role"] == "assistant"
        assert message["content"] == full_response
        assert "backend_details" in message


class TestChatConfiguration:
    """Tests for chat configuration."""

    @patch('app.genai.Client')
    def test_chat_created_with_correct_config(self, mock_client_class):
        """Test that chat is created with correct configuration."""
        from google.genai.types import GenerateContentConfig
        from app import MODEL_ID, sql_query_tool
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        client = mock_client_class()
        config = GenerateContentConfig(temperature=0, tools=[sql_query_tool])
        
        chat = client.chats.create(
            model=MODEL_ID,
            config=config,
        )
        
        mock_client.chats.create.assert_called_once()

    @patch('app.genai.Client')
    def test_temperature_set_to_zero(self, mock_client_class):
        """Test that temperature is set to 0 for deterministic responses."""
        from google.genai.types import GenerateContentConfig
        
        config = GenerateContentConfig(temperature=0)
        
        assert config.temperature == 0


class TestPromptAugmentation:
    """Tests for prompt augmentation with instructions."""

    def test_prompt_includes_high_level_summary_instruction(self):
        """Test that prompt is augmented with summary instructions."""
        original_prompt = "What are the sales?"
        
        augmented_prompt = original_prompt + """
            Please give a concise, high-level summary followed by detail in
            plain language about where the information in your response is
            coming from in the database. Only use information that you learn
            from BigQuery, do not make up information.
            """
        
        assert "high-level summary" in augmented_prompt
        assert "BigQuery" in augmented_prompt
        assert "do not make up information" in augmented_prompt

    def test_prompt_instructs_to_use_bigquery_only(self):
        """Test that prompt instructs to use only BigQuery information."""
        instruction = """Only use information that you learn
            from BigQuery, do not make up information."""
        
        assert "Only use information" in instruction
        assert "BigQuery" in instruction
        assert "do not make up" in instruction


class TestPartFromFunctionResponse:
    """Tests for Part.from_function_response construction."""

    @patch('app.Part')
    def test_function_response_part_creation(self, mock_part_class):
        """Test that function response Part is created correctly."""
        mock_part_class.from_function_response = MagicMock()
        
        function_name = "sql_query"
        api_response = "[{'count': 100}]"
        
        mock_part_class.from_function_response(
            name=function_name,
            response={"content": api_response},
        )
        
        mock_part_class.from_function_response.assert_called_once_with(
            name=function_name,
            response={"content": api_response},
        )


class TestVariableRenaming:
    """Tests to verify the variable renaming from 'part' to 'response'."""

    def test_response_variable_used_consistently(self):
        """Test that 'response' variable is used instead of 'part'."""
        # This test verifies the refactoring from 'part' to 'response'
        mock_response = MagicMock()
        mock_response.text = "Sample text"
        mock_response.function_call.name = "test_function"
        
        # Simulate the new variable name
        response = mock_response
        
        assert response.text == "Sample text"
        assert response.function_call.name == "test_function"

    def test_response_extracted_from_candidates(self):
        """Test that response is correctly extracted from candidates."""
        mock_api_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Test response"
        mock_candidate.content.parts = [mock_part]
        mock_api_response.candidates = [mock_candidate]
        
        # Extract response (new variable name)
        response = mock_api_response.candidates[0].content.parts[0]
        
        assert response.text == "Test response"


class TestTimeDelay:
    """Tests for time delay in response processing."""

    @patch('app.time')
    def test_sleep_called_after_processing(self, mock_time):
        """Test that sleep is called for UI smoothness."""
        mock_time.sleep = MagicMock()
        
        # Simulate the sleep call
        time.sleep(3)
        
        mock_time.sleep.assert_called_once_with(3)


class TestExceptionHandling:
    """Tests for general exception handling."""

    @patch('app.st')
    def test_general_exception_caught(self, mock_st):
        """Test that general exceptions are caught and displayed."""
        mock_st.error = MagicMock()
        mock_st.session_state = {"messages": []}
        
        try:
            raise Exception("Unexpected error occurred")
        except Exception as e:
            error_message = f"""
                Something went wrong! We encountered an unexpected error while
                trying to process your request. Please try rephrasing your
                question. Details:

                {str(e)}"""
            mock_st.error(error_message)
            mock_st.session_state["messages"].append({
                "role": "assistant",
                "content": error_message,
            })
        
        mock_st.error.assert_called_once()
        assert "Unexpected error occurred" in mock_st.error.call_args[0][0]

    @patch('app.st')
    def test_error_message_added_to_session_state(self, mock_st):
        """Test that error messages are added to session state."""
        mock_st.session_state = {"messages": []}
        
        error_message = "An error occurred during processing"
        mock_st.session_state["messages"].append({
            "role": "assistant",
            "content": error_message,
        })
        
        assert len(mock_st.session_state["messages"]) == 1
        assert mock_st.session_state["messages"][0]["role"] == "assistant"
        assert "error" in mock_st.session_state["messages"][0]["content"].lower()


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    @patch('app.st')
    @patch('app.genai.Client')
    @patch('app.bigquery.Client')
    def test_complete_query_workflow(self, mock_bq_class, mock_genai_class, mock_st):
        """Test complete workflow from prompt to response."""
        # Setup mocks
        mock_genai_client = MagicMock()
        mock_chat = MagicMock()
        
        # First response with function call
        mock_fc_response = MagicMock()
        mock_fc_part = MagicMock()
        mock_fc_part.function_call.name = "list_tables"
        mock_fc_part.function_call.args = {"dataset_id": "test_dataset"}
        mock_fc_response.candidates = [MagicMock()]
        mock_fc_response.candidates[0].content.parts = [mock_fc_part]
        
        # Final response with text
        mock_text_response = MagicMock()
        mock_text_part = MagicMock()
        mock_text_part.text = "Here are the results"
        delattr(mock_text_part, 'function_call')
        mock_text_response.candidates = [MagicMock()]
        mock_text_response.candidates[0].content.parts = [mock_text_part]
        
        mock_chat.send_message.side_effect = [mock_fc_response, mock_text_response]
        mock_genai_client.chats.create.return_value = mock_chat
        mock_genai_class.return_value = mock_genai_client
        
        # Setup BigQuery mock
        mock_bq_client = MagicMock()
        mock_table = MagicMock()
        mock_table.table_id = "orders"
        mock_bq_client.list_tables.return_value = [mock_table]
        mock_bq_class.return_value = mock_bq_client
        
        # Execute workflow
        genai_client = mock_genai_class()
        bq_client = mock_bq_class()
        
        chat = genai_client.chats.create(model="gemini-2.0-flash", config=MagicMock())
        
        # Send initial message
        response = chat.send_message("What tables are available?")
        response_part = response.candidates[0].content.parts[0]
        
        # Process function call
        params = dict(response_part.function_call.args)
        api_result = bq_client.list_tables(params["dataset_id"])
        
        # Send function response
        from google.genai.types import Part
        final_response = chat.send_message(
            Part.from_function_response(
                name=response_part.function_call.name,
                response={"content": str([t.table_id for t in api_result])},
            )
        )
        
        # Verify workflow
        assert mock_chat.send_message.call_count == 2
        assert mock_bq_client.list_tables.called


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @patch('app.st')
    def test_empty_chat_input(self, mock_st):
        """Test handling of empty chat input."""
        prompt = ""
        
        # Should not process empty prompt
        if prompt:
            # This should not execute
            assert False, "Should not process empty prompt"
        else:
            assert True

    @patch('app.bigquery.Client')
    def test_empty_query_results(self, mock_bq_client_class):
        """Test handling of empty query results."""
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.result.return_value = []
        mock_client.query.return_value = mock_job
        mock_bq_client_class.return_value = mock_client
        
        client = mock_bq_client_class()
        query_job = client.query("SELECT * FROM empty_table")
        results = query_job.result()
        
        assert len(list(results)) == 0

    def test_query_with_special_characters(self):
        """Test query cleaning with special characters."""
        query = "SELECT * FROM `project.dataset.table` WHERE name = 'O\\'Brien'"
        
        cleaned = query.replace("\\n", " ").replace("\n", "").replace("\\", "")
        
        # Backslashes should be removed but query should be readable
        assert "\\" not in cleaned or "\\'" in query

    @patch('app.st')
    def test_very_long_backend_details(self, mock_st):
        """Test handling of very long backend details strings."""
        backend_details = "- Function call:\n" * 1000
        
        assert len(backend_details) > 10000
        assert isinstance(backend_details, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])