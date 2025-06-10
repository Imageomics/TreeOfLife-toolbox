def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption("--root", action="store", help="Root directory for parquet files.")

    parser.addoption("--split", action="store", help="Real split.")
