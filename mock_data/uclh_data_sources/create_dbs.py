import sqlite3
import pandas as pd
import os
from pathlib import Path

# Define mock database directory
MOCK_DB_DIR = Path(".")
MOCK_DB_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


def get_mock_database_file(db_name):
    return MOCK_DB_DIR / f"mock_{db_name}.sqlite"

# Mapping col_types to pandas dtype
col_type_map = {
    "i": "Int64",  # Nullable integer type (Int64 instead of int64)
    "c": "object",  # For string data
    "d": "float64",
    "T": "datetime64",  # If 'T' represents datetime
}

# Function to check if mock database is up to date
def check_up_to_date():
    files = list(MOCK_DB_DIR.glob("**/*"))
    source_files = [f for f in files if not f.suffix == ".sqlite"]
    output_files = [f for f in files if f.suffix == ".sqlite"]
    
    if source_files and output_files:
        latest_source_time = max(f.stat().st_mtime for f in source_files)
        earliest_output_time = min(f.stat().st_mtime for f in output_files)
        
        if latest_source_time > earliest_output_time:
            print("Please run this script again to recreate the databases.")

# Function to attach other databases as schemas
def attach_file_as_schema(con, schema):
    schema_file = get_mock_database_file(schema)
    con.execute(f"ATTACH DATABASE '{schema_file}' AS {schema};")

# Function to connect to mock databases
def connect_to_mock_caboodle():
    check_up_to_date()
    
    con = sqlite3.connect(get_mock_database_file("dbo"))
    for schema in ["EPIC", "REF", "Ext", "UCLH", "integration"]:
        attach_file_as_schema(con, schema)
    
    return con

# Function to remove old/existing databases
def remove_databases():
    for file in MOCK_DB_DIR.glob("*.sqlite"):
        file.unlink()

# Function to create and populate a database from DDL and CSV
def build_database(db_name):
    db_path = get_mock_database_file(db_name)

    if db_path.exists():
        print(f"Database {db_path} already exists. Skipping creation.")
        return  # Skip creation if DB exists

    print(f"Creating database: {db_path}")

    con = sqlite3.connect(db_path)
    prefix = db_name.lower()

    # Execute all DDL files
    ddl_dir = MOCK_DB_DIR / f"{prefix}_ddl"
    if ddl_dir.exists():
        for ddl_file in ddl_dir.glob("*.sql"):
            with open(ddl_file, "r") as f:
                con.executescript(f.read())

    # Load all CSV data
    col_types_path = MOCK_DB_DIR / "col_types.csv"
    if col_types_path.exists():
        col_types_df = pd.read_csv(col_types_path, dtype=str)
        csv_dir = MOCK_DB_DIR / f"{prefix}_data"

        if csv_dir.exists():
            for csv_file in csv_dir.glob("*.csv"):
                table_name = csv_file.stem
                col_types_row = col_types_df[col_types_df["table"] == table_name]

                if not col_types_row.empty:
                    # Get the col_types string for this table
                    col_types_string = col_types_row["col_types"].values[0]

                    # Generate a dtype_map assuming columns in the CSV follow the col_types in order
                    column_names = pd.read_csv(csv_file, nrows=0).columns  # Get column names from the first row

                    if len(column_names) == len(col_types_string):
                        # Map each column to the corresponding pandas dtype
                        dtype_map = {column: col_type_map[ctype] for column, ctype in zip(column_names, col_types_string)}

                        # Identify columns that should be parsed as dates (where 'T' is used in col_types)
                        parse_dates = [column for column, ctype in zip(column_names, col_types_string) if ctype == "T"]

                        # Remove datetime columns from dtype_map as they're being parsed separately
                        for date_column in parse_dates:
                            dtype_map.pop(date_column, None)

                        # Load the data with the dtype map and parse_dates
                        data = pd.read_csv(csv_file, dtype=dtype_map, parse_dates=parse_dates)
                        print(f"Loading mock data from: {csv_file.name}")
                        data.to_sql(table_name, con, if_exists="replace", index=False)

    con.close()

# Function to recreate all mock databases
def recreate_mock_database():
    remove_databases()

    database_list = ["dbo", "EPIC", "REF", "Ext", "UCLH", "integration"]
    for db in database_list:
        build_database(db)

# Entry point
if __name__ == "__main__":
    print("Recreating mock databases...")
    recreate_mock_database()
    print("Mock databases have been recreated successfully.")
