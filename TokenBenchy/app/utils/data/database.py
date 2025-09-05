from __future__ import annotations

import os
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import Column, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import declarative_base, sessionmaker
from tqdm import tqdm

from TokenBenchy.app.constants import DATA_PATH
from TokenBenchy.app.logger import logger
from TokenBenchy.app.utils.singleton import singleton

Base = declarative_base()


###############################################################################
class BenchmarkResults(Base):
    __tablename__ = "BENCHMARK_RESULTS"
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
    __table_args__ = (UniqueConstraint("tokenizer", "text"),)


###############################################################################
class NSLBenchmark(Base):
    __tablename__ = "NSL_RESULTS"
    tokenizer = Column(String, primary_key=True)
    tokens_count = Column(Integer)
    __table_args__ = (UniqueConstraint("tokenizer"),)


###############################################################################
class VocabularyStatistics(Base):
    __tablename__ = "VOCABULARY_STATISTICS"
    tokenizer = Column(String, primary_key=True)
    number_tokens_from_vocabulary = Column(Integer)
    number_tokens_from_decode = Column(Integer)
    number_shared_tokens = Column(Integer)
    number_unshared_tokens = Column(Integer)
    percentage_subwords = Column(Float)
    percentage_true_words = Column(Float)
    __table_args__ = (UniqueConstraint("tokenizer"),)


###############################################################################
class Vocabulary(Base):
    __tablename__ = "VOCABULARY"
    tokenizer = Column(String, primary_key=True)
    token_id = Column(Integer, primary_key=True)
    vocabulary_tokens = Column(String)
    decoded_tokens = Column(String)
    __table_args__ = (UniqueConstraint("tokenizer", "token_id"),)


###############################################################################
class TextDataset(Base):
    __tablename__ = "TEXT_DATASET"
    dataset_name = Column(String, primary_key=True)
    text = Column(String, primary_key=True)
    __table_args__ = (UniqueConstraint("dataset_name", "text"),)


###############################################################################
class TextDatasetStatistics(Base):
    __tablename__ = "TEXT_DATASET_STATISTICS"
    dataset_name = Column(String, primary_key=True)
    text = Column(String, primary_key=True)
    words_count = Column(Integer)
    AVG_words_length = Column(Float)
    STD_words_length = Column(Float)
    __table_args__ = (UniqueConstraint("dataset_name", "text"),)


# [DATABASE]
###############################################################################
@singleton
class TokenBenchyDatabase:
    def __init__(self) -> None:
        self.db_path = os.path.join(DATA_PATH, "TokenBenchy_database.db")
        self.engine = create_engine(
            f"sqlite:///{self.db_path}", echo=False, future=True
        )
        self.Session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = 2000

    # -------------------------------------------------------------------------
    def initialize_database(self) -> None:
        Base.metadata.create_all(self.engine)

    # -------------------------------------------------------------------------
    def get_table_class(self, table_name: str) -> Any:
        for cls in Base.__subclasses__():
            if hasattr(cls, "__tablename__") and cls.__tablename__ == table_name:
                return cls
        raise ValueError(f"No table class found for name {table_name}")

    # -------------------------------------------------------------------------
    def _upsert_dataframe(
        self, df: pd.DataFrame, table_cls: Any, batch_size: int | None = None
    ) -> None:
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
            records = df.to_dict(orient="records")
            for i in tqdm(
                range(0, len(records), batch_size), desc="[INFO] Updating database"
            ):
                batch = records[i : i + batch_size]
                stmt = insert(table).values(batch)
                # Columns to update on conflict
                update_cols = {
                    c: getattr(stmt.excluded, c)  # type: ignore
                    for c in batch[0]
                    if c not in unique_cols
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols, set_=update_cols
                )
                session.execute(stmt)
                session.commit()
            session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            data = pd.read_sql_table(table_name, conn)

        return data

    # -------------------------------------------------------------------------
    def save_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(sqlalchemy.text(f'DELETE FROM "{table_name}"'))
            df.to_sql(table_name, conn, if_exists="append", index=False)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        table_cls = self.get_table_class(table_name)
        self._upsert_dataframe(df, table_cls)

    # -------------------------------------------------------------------------
    def export_all_tables_as_csv(self, chunksize: int | None = None) -> None:
        export_path = os.path.join(DATA_PATH, "export")
        os.makedirs(export_path, exist_ok=True)
        with self.engine.connect() as conn:
            for table in Base.metadata.sorted_tables:
                table_name = table.name
                csv_path = os.path.join(export_path, f"{table_name}.csv")

                # Build a safe SELECT for arbitrary table names (quote with "")
                query = sqlalchemy.text(f'SELECT * FROM "{table_name}"')

                if chunksize:
                    first = True
                    for chunk in pd.read_sql(query, conn, chunksize=chunksize):
                        chunk.to_csv(
                            csv_path,
                            index=False,
                            header=first,
                            mode="w" if first else "a",
                            encoding="utf-8",
                            sep=",",
                        )
                        first = False
                    # If table is empty, still write header row
                    if first:
                        pd.DataFrame(columns=[c.name for c in table.columns]).to_csv(
                            csv_path, index=False, encoding="utf-8", sep=","
                        )
                else:
                    df = pd.read_sql(query, conn)
                    if df.empty:
                        pd.DataFrame(columns=[c.name for c in table.columns]).to_csv(
                            csv_path, index=False, encoding="utf-8", sep=","
                        )
                    else:
                        df.to_csv(csv_path, index=False, encoding="utf-8", sep=",")

        logger.info(f"All tables exported to CSV at {os.path.abspath(export_path)}")

    # -------------------------------------------------------------------------
    def delete_all_data(self) -> None:
        with self.engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(table.delete())


# -----------------------------------------------------------------------------
database = TokenBenchyDatabase()
