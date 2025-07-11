import os
import re
import pandas as pd
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.dialects.sqlite import insert

from TokenBenchy.app.src.constants import DATA_PATH
from TokenBenchy.app.src.logger import logger

Base = declarative_base()

###############################################################################
class BenchmarkResults(Base):
    __tablename__ = 'BENCHMARK_RESULTS'
    tokenizer = Column(String, primary_key=True)
    num_characters = Column(Integer)
    words_count = Column(Integer)
    AVG_words_length = Column(Float)
    tokens_count = Column(Integer)
    tokens_characters = Column(Integer)
    AVG_tokens_length = Column(Float)
    tokens_to_words_ratio = Column(Float)
    bytes_per_token = Column(Float)
    __table_args__ = (
        UniqueConstraint('tokenizer'),
    )

###############################################################################
class VocabularyStatistics(Base):
    __tablename__ = 'VOCABULARY_STATISTICS'
    tokenizer = Column(String, primary_key=True)
    number_tokens_from_vocabulary = Column(Integer)
    number_tokens_from_decode = Column(Integer)
    number_shared_tokens = Column(Integer)
    number_unshared_tokens = Column(Integer)
    percentage_subwords = Column(Float)
    percentage_true_words = Column(Float)
    __table_args__ = (
        UniqueConstraint('tokenizer'),
    )


###############################################################################
class Vocabulary(Base):
    __tablename__ = 'VOCABULARY'
    id = Column(Integer, primary_key=True)
    vocabulary_tokens = Column(String, primary_key=True)
    decoded_tokens = Column(String)
    __table_args__ = (
        UniqueConstraint('id', 'vocabulary_tokens'),
    )


###############################################################################
class DatasetStatistics(Base):
    __tablename__ = 'DATASET_STATISTICS'
    text = Column(String, primary_key=True)
    words_count = Column(Integer)
    AVG_word_length = Column(Float)
    STD_word_length = Column(Float)
    __table_args__ = (
        UniqueConstraint('text'),
    )

# [DATABASE]
###############################################################################
class TokenBenchyDatabase:

    def __init__(self):                   
        self.db_path = os.path.join(DATA_PATH, 'TokenBenchy_database.db')
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False, future=True)
        self.Session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = 10000        
       
    #--------------------------------------------------------------------------       
    def initialize_database(self): 
        Base.metadata.create_all(self.engine)

    #--------------------------------------------------------------------------
    def upsert_dataframe(self, df: pd.DataFrame, table_cls):
        table = table_cls.__table__
        session = self.Session()
        try:
            unique_cols = []
            for uc in table.constraints:
                if isinstance(uc, UniqueConstraint):
                    unique_cols = uc.columns.keys()
                    break
            if not unique_cols:
                raise ValueError(f"No unique constraint found for {table_cls.__name__}")

            # Batch insertions for speed
            records = df.to_dict(orient='records')
            for i in range(0, len(records), self.insert_batch_size):
                batch = records[i:i + self.insert_batch_size]
                stmt = insert(table).values(batch)
                # Columns to update on conflict
                update_cols = {c: getattr(stmt.excluded, c) for c in batch[0] if c not in unique_cols}
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols,
                    set_=update_cols
                )
                session.execute(stmt)
            session.commit()
        finally:
            session.close()       

    #--------------------------------------------------------------------------
    def load_benchmark_results(self):            
        with self.engine.connect() as conn:
            benchmarks = pd.read_sql_table("BENCHMARK_RESULTS", conn)
            stats = pd.read_sql_table("VOCABULARY_STATISTICS", conn)

        return benchmarks, stats

    #--------------------------------------------------------------------------
    def load_vocabulary_tokens(self, table_name=None):
        table_name = re.sub(r'[^0-9A-Za-z_]', '_', table_name) if table_name else None
        table_name = "VOCABULARY" if table_name is None else f'{table_name}_VOCABULARY'
        with self.engine.connect() as conn:
            vocabulary = pd.read_sql_table(table_name, conn)
        return vocabulary

    #--------------------------------------------------------------------------
    def save_dataset_statistics(self, data):         
        with self.engine.begin() as conn:
            data.to_sql("DATASET_STATISTICS", conn, if_exists='replace', index=False)

    #--------------------------------------------------------------------------
    def save_benchmark_results(self, data: pd.DataFrame, table_name=None):
        table_name = re.sub(r'[^0-9A-Za-z_]', '_', table_name) if table_name else None
        table_name = "BENCHMARK_RESULTS" if table_name is None else f'{table_name}_BENCHMARK_RESULTS'
        with self.engine.begin() as conn:
            data.to_sql(table_name, conn, if_exists='replace', index=False)

    #--------------------------------------------------------------------------
    def save_vocabulary_results(self, data: pd.DataFrame):
        with self.engine.begin() as conn:
            data.to_sql("VOCABULARY_STATISTICS", conn, if_exists='replace', index=False)

    #--------------------------------------------------------------------------
    def save_vocabulary_tokens(self, data: pd.DataFrame, table_name=None):
        table_name = re.sub(r'[^0-9A-Za-z_]', '_', table_name) if table_name else None
        table_name = "VOCABULARY" if table_name is None else f'{table_name}_VOCABULARY'
        with self.engine.begin() as conn:
            data.to_sql(table_name, conn, if_exists='replace', index=False)
    

   

 
    
    