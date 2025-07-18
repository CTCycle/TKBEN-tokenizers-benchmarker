import os
import re
import pandas as pd
import sqlalchemy
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.dialects.sqlite import insert
from tqdm import tqdm

from TokenBenchy.app.constants import DATA_PATH
from TokenBenchy.app.logger import logger

Base = declarative_base()

###############################################################################
class BenchmarkResults(Base):
    __tablename__ = 'BENCHMARK_RESULTS'
    tokenizer = Column(String, primary_key=True)
    text = Column(String, primary_key=True)
    num_characters = Column(Integer)
    words_count = Column(Integer)
    AVG_words_length = Column(Float)
    tokens_count = Column(Integer)
    tokens_characters = Column(Integer)
    AVG_tokens_length = Column(Float)
    tokens_to_words_ratio = Column(Float)
    bytes_per_token = Column(Float)
    __table_args__ = (
        UniqueConstraint('tokenizer', 'text'),
    )

###############################################################################
class NSLBenchmark(Base):
    __tablename__ = 'NSL_RESULTS'
    tokenizer = Column(String, primary_key=True)
    tokens_count = Column(Integer)
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
    tokenizer = Column(String, primary_key=True)
    token_id = Column(Integer, primary_key=True)
    vocabulary_tokens = Column(String)
    decoded_tokens = Column(String)
    __table_args__ = (
        UniqueConstraint('tokenizer', 'token_id'),
    )


###############################################################################
class TextDataset(Base):
    __tablename__ = 'TEXT_DATASET'
    dataset_name = Column(String, primary_key=True)
    text = Column(String, primary_key=True)
    words_count = Column(Integer)
    AVG_words_length = Column(Float)
    STD_words_length = Column(Float)
    __table_args__ = (
        UniqueConstraint('dataset_name', 'text'),
    )


# [DATABASE]
###############################################################################
class TokenBenchyDatabase:

    def __init__(self):                   
        self.db_path = os.path.join(DATA_PATH, 'TokenBenchy_database.db')
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False, future=True)
        self.Session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = 20000     
       
    #--------------------------------------------------------------------------       
    def initialize_database(self): 
        Base.metadata.create_all(self.engine)

    #--------------------------------------------------------------------------
    def upsert_dataframe(self, df: pd.DataFrame, table_cls, batch_size=None):
        batch_size = batch_size if batch_size else self.insert_batch_size
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
            for i in tqdm(range(0, len(records), batch_size), desc=f'[INFO] Updating database'):
                batch = records[i:i + batch_size]
                stmt = insert(table).values(batch)
                # Columns to update on conflict
                update_cols = {c: getattr(stmt.excluded, c) for c in batch[0] if c not in unique_cols}
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols,
                    set_=update_cols
                )
                session.execute(stmt)
                session.commit()
            session.commit()
        finally:
            session.close()  

    #--------------------------------------------------------------------------
    def load_text_dataset(self):            
        with self.engine.connect() as conn:
            text_dataset = pd.read_sql_table("TEXT_DATASET", conn)

        return text_dataset   

    #--------------------------------------------------------------------------
    def load_benchmark_results(self):            
        with self.engine.connect() as conn:
            benchmarks = pd.read_sql_table("BENCHMARK_RESULTS", conn)
            stats = pd.read_sql_table("VOCABULARY_STATISTICS", conn)

        return benchmarks, stats

    #--------------------------------------------------------------------------
    def load_vocabularies(self):        
        with self.engine.connect() as conn:
            vocabulary = pd.read_sql_table('VOCABULARY', conn)
        return vocabulary
    
    #--------------------------------------------------------------------------
    def save_text_dataset(self, data : pd.DataFrame):
        with self.engine.begin() as conn:
            conn.execute(sqlalchemy.text(f"DELETE FROM TEXT_DATASET"))         
        data.to_sql('TEXT_DATASET', self.engine, if_exists='append', index=False)

    #--------------------------------------------------------------------------
    def save_dataset_statistics(self, data : pd.DataFrame):         
        self.upsert_dataframe(data, TextDataset)

    #--------------------------------------------------------------------------
    def save_benchmark_results(self, data: pd.DataFrame):
        self.upsert_dataframe(data, BenchmarkResults)

    #--------------------------------------------------------------------------
    def save_NSL_benchmark(self, data: pd.DataFrame):
        with self.engine.begin() as conn:
            conn.execute(sqlalchemy.text(f"DELETE FROM NSL_RESULTS"))        
        data.to_sql('NSL_RESULTS', self.engine, if_exists='append', index=False) 
    
    #--------------------------------------------------------------------------
    def save_vocabulary_statistics(self, data: pd.DataFrame):
        with self.engine.begin() as conn:
            conn.execute(sqlalchemy.text(f"DELETE FROM VOCABULARY_STATISTICS"))        
        data.to_sql('VOCABULARY_STATISTICS', self.engine, if_exists='append', index=False)       

    #--------------------------------------------------------------------------
    def save_vocabulary_tokens(self, data: pd.DataFrame):
        self.upsert_dataframe(data, Vocabulary)
    

   

 
    
    