"""
Database Schema Initialization Script

This script initializes the ClickHouse database schema by executing SQL statements
from schema.sql. It handles comment removal and statement parsing for clean execution.

This is typically run once during initial setup or when schema changes are needed.
"""
import clickhouse_connect
import re
from dotenv import load_dotenv
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

logger.info("Starting database schema initialization")

# Load environment variables
load_dotenv()

# ============================================================================
# ClickHouse Connection
# ============================================================================
# Connect to ClickHouse database for schema initialization
CLICKHOUSE_PORT = os.getenv("CLICKHOUSE_PORT")
if not CLICKHOUSE_PORT:
    logger.error("CLICKHOUSE_PORT environment variable not set")
    raise ValueError("CLICKHOUSE_PORT environment variable is required")

try:
    CLICKHOUSE_PORT = int(CLICKHOUSE_PORT)
    logger.info(f"Connecting to ClickHouse on localhost:{CLICKHOUSE_PORT}")
    
    client = clickhouse_connect.get_client(
        host='localhost',
        port=CLICKHOUSE_PORT,
        username='default',
    )
    logger.info("Successfully connected to ClickHouse")
except Exception as e:
    logger.error(f"Failed to connect to ClickHouse: {str(e)}", exc_info=True)
    raise

# ============================================================================
# SQL Schema File Processing
# ============================================================================
# Read and parse SQL schema file, removing comments and splitting into statements
schema_file = "schema.sql"
if not Path(schema_file).exists():
    logger.error(f"Schema file not found: {schema_file}")
    raise FileNotFoundError(f"Schema file not found: {schema_file}")

logger.info(f"Reading schema file: {schema_file}")
try:
    with open(schema_file, 'r') as f:
        sql = f.read()
    logger.info(f"Schema file read successfully ({len(sql)} characters)")
except Exception as e:
    logger.error(f"Failed to read schema file: {str(e)}", exc_info=True)
    raise

# Remove SQL comments to avoid parsing issues
# Remove multiline comments: /* ... */
sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.S)
logger.debug("Removed multiline comments from SQL")

# Remove single line comments: --
sql = re.sub(r'--.*?$', '', sql, flags=re.M)
logger.debug("Removed single-line comments from SQL")

# Split SQL into individual statements (separated by semicolons)
statements = [s.strip() for s in sql.split(';') if s.strip()]
logger.info(f"Parsed {len(statements)} SQL statements from schema file")

# ============================================================================
# Execute SQL Statements
# ============================================================================
# Execute each SQL statement sequentially
# Stop on first error to prevent partial schema initialization
logger.info("Executing SQL statements...")
for idx, stmt in enumerate(statements, 1):
    # Log statement preview (first 80 characters)
    stmt_preview = stmt[:80] + "..." if len(stmt) > 80 else stmt
    logger.info(f"Executing statement {idx}/{len(statements)}: {stmt_preview}")
    print(f"\nExecuting statement {idx}/{len(statements)}: {stmt_preview}")
    
    try:
        client.command(stmt)
        logger.debug(f"Statement {idx} executed successfully")
    except Exception as e:
        logger.error(
            f"Error executing statement {idx}: {str(e)}\n"
            f"Statement: {stmt_preview}",
            exc_info=True
        )
        print(f"❌ ERROR in statement {idx}: {stmt_preview}")
        print(f"Error: {e}")
        # Break on error to prevent partial schema initialization
        break

logger.info("Database schema initialization completed successfully")
print("\n✅ Schema initialization completed!")
