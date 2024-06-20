from etl import defs


def test_defs():
    """Test that defs contains the required JobDefinitions."""
    assert defs.get_job_def("embedding_job")