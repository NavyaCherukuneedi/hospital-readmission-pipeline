"""Database connection and utilities."""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config/.env')

class DatabaseConnection:
    """Manage PostgreSQL database connections."""
    
    def __init__(self):
        """Initialize database connection."""
        self.db_url = os.getenv('DB_URL')
        if not self.db_url:
            raise ValueError("DB_URL not found in environment variables")
        
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
    
    def test_connection(self):
        """Test if database connection works."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def create_table(self, table_name, df):
        """Create table from DataFrame schema."""
        df.head(0).to_sql(table_name, self.engine, if_exists='replace', index=False)
    
    def insert_data(self, table_name, df, chunksize=1000):
        """Insert DataFrame into table."""
        df.to_sql(table_name, self.engine, if_exists='append', index=False, chunksize=chunksize)
    
    def query(self, sql):
        """Execute a SQL query and return results."""
        with self.engine.connect() as connection:
            result = connection.execute(text(sql))
            return result.fetchall()
    
    def close(self):
        """Close database connection."""
        self.engine.dispose()