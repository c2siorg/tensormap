import unittest
from unittest.mock import patch, MagicMock, call
from endpoints.DataProcess.services import (
    add_target_service,
    generic_response,
    save_one_record,
    DataFile,
    DataProcess,
)

FILE_ID = "file_id"
FILE_TARGET_FIELD = "target"


class TestAddTargetService(unittest.TestCase):
    @patch("endpoints.DataProcess.services.DataFile")
    def test_add_target_service_file_not_found(self, mock_datafile):
        """
        Test case: File does not exist in the database.
        """
        mock_filter_by = MagicMock()
        mock_filter_by.count.return_value = 0
        mock_query = MagicMock()
        mock_query.filter_by.return_value = mock_filter_by
        mock_datafile.query = mock_query

        response = add_target_service({FILE_ID: 999, FILE_TARGET_FIELD: "target"})

        self.assertEqual(
            response,
            generic_response(
                status_code=400, success=False, message="File doesn't exist in DB"
            ),
        )

    @patch("endpoints.DataProcess.services.save_one_record")
    @patch("endpoints.DataProcess.services.DataProcess")
    @patch("endpoints.DataProcess.services.DataFile")
    def test_add_target_service_success(
        self, mock_datafile, mock_dataprocess, mock_save_one_record
    ):
        """
        Test case: File exists, and the target field is added successfully.
        """
        mock_file_instance = MagicMock()
        mock_filter_by = MagicMock()
        mock_filter_by.count.return_value = 1
        mock_filter_by.first.return_value = mock_file_instance
        mock_query = MagicMock()
        mock_query.filter_by.return_value = mock_filter_by
        mock_datafile.query = mock_query

        mock_dataprocess_instance = MagicMock()
        mock_dataprocess.return_value = mock_dataprocess_instance

        response = add_target_service({FILE_ID: 123, FILE_TARGET_FIELD: "target"})

        self.assertEqual(
            response,
            generic_response(
                status_code=201, success=True, message="Target field added successfully"
            ),
        )

        mock_dataprocess.assert_called_once_with(
            file_id=123, file=mock_file_instance, target="target"
        )

        mock_save_one_record.assert_called_once_with(record=mock_dataprocess_instance)

    @patch("endpoints.DataProcess.services.DataFile")
    def test_add_target_service_exception(self, mock_datafile):
        """
        Test case: An exception occurs during processing.
        """
        mock_query = MagicMock()
        mock_query.filter_by.side_effect = Exception("Database error")
        mock_datafile.query = mock_query

        response = add_target_service({FILE_ID: 456, FILE_TARGET_FIELD: "target"})

        self.assertEqual(
            response,
            generic_response(
                status_code=500,
                success=False,
                message="Error storing record: Database error",
            ),
        )


if __name__ == "__main__":
    unittest.main()
