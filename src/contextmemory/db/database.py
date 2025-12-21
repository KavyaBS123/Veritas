from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base 

from src.contextmemory.core.config import settings

try:
    engine = create_engine(
        settings.DATABASE_URL
    )
    with engine.connect() as conn:
        print("Database connected sucessfully")
    
except Exception as e: 
    print("Failed to connect database")
    raise e


SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_table():
    try:
        import src.contextmemory.db.models

        Base.metadata.create_all(bind=engine)
        print("Tables created successfully")

    except Exception as e:
        print("Error while creating tables")
        raise e
