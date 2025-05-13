from sqlalchemy import *
from sqlalchemy.orm import relationship, declarative_base, Mapped, mapped_column, Session
from typing import *
import sys
from argparse import Namespace
import os

"""
Classes for managing SQL database for storing experiment results.

Adapted from Gemini output on May 8th 2025.
Original Gemini prompt:
```
I'm uploading an image of a SQL database schema. Write me a file with SQL alchemy classes for each table in this schema.
```
Schema stored at: https://drawsql.app/teams/ucsd-6/diagrams/ml-experiments
"""

SQL_PREFIX = "sqlite:///"
Base = declarative_base()

class Model(Base):
    """
    - id: int
    - name: str
    - base: int
    - peft: str
    """
    __tablename__ = 'model'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[String] = mapped_column(String(20), unique=True)
    base = Column(Integer, ForeignKey("model.id"), nullable=True)
    peft = Column(String(20))
    
    base_model = relationship("Model", remote_side=[id])
    experiments = relationship("Experiment", back_populates="model")

class Dataset(Base):
    """
    - id: int
    - language: str
    - name: str
    - split: str
    - experiments: List[int]
    """
    __tablename__ = 'dataset'
    id: Mapped[int] = mapped_column(primary_key=True)
    language = Column(String(20))
    name = Column(String(20))  
    split = Column(String(10))
    experiments = relationship("Experiment", secondary='experiment_data', back_populates="datasets")


class Experiment(Base):
    """
    - id: int
    - argv: str
    - description: str
    - datasets: List[int]
    - model: int
    - events: List[int]
    - hyperparams: List[int]
    """
    __tablename__ = 'experiment'
    id: Mapped[int] = mapped_column(primary_key=True)
    argv = Column(UnicodeText)
    description = Column(UnicodeText)
    model_id = mapped_column(Integer, ForeignKey('model.id'))

    datasets = relationship("Dataset", secondary="experiment_data", back_populates="experiments")
    model = relationship("Model", back_populates="experiments")
    events = relationship('Event', back_populates="experiment")
    hyperparams = relationship('Hyperparam', back_populates="experiment")

class ExperimentData(Base):
    """
    - id: int
    - experiment_id: int
    - dataset_id: int
    """
    __tablename__ = 'experiment_data'
    id = Column(Integer, primary_key=True)
    experiment_id=Column(ForeignKey('experiment.id'))
    dataset_id = Column(ForeignKey('dataset.id'))

class Event(Base):
    """
    - id: int
    - tag: str
    - value: float
    - eval_params: List[int]
    - experiment: int
    """
    __tablename__ = 'event'
    id: Mapped[int] = mapped_column(primary_key=True)
    tag = Column(String(20))
    value = Column(Float())

    experiment_id = mapped_column(Integer, ForeignKey('experiment.id'))

    eval_params = relationship("EvalParam", secondary="event_param", back_populates="events")
    experiment = relationship("Experiment", back_populates="events")

class EvalParam(Base):
    """
    - id: int
    - param: str
    - value: float
    - text: str
    - events: List[str]
    """
    __tablename__ = 'eval_param'
    id: Mapped[int] = mapped_column(primary_key=True)
    param = Column(String(20))
    value = Column(Float())
    text = Column(UnicodeText())

    events = relationship("Event", secondary="event_param", back_populates="eval_params")

class EventParam(Base):
    """
    - id: int
    - event_id: int
    - param_id: int
    """
    __tablename__ = 'event_param'
    id: Mapped[int] = mapped_column(primary_key=True)
    event_id = Column(ForeignKey('event.id'))
    param_id = Column(ForeignKey('eval_param.id'))

class Hyperparam(Base):
    """
    - id: int
    - param: str
    - text: str
    - experiment: int
    """
    __tablename__ = 'hyperparam'
    id: Mapped[int] = mapped_column(primary_key=True)
    param = Column(String(20))
    value = Column(Float())
    text = Column(String(20))
    experiment_id = mapped_column(Integer, ForeignKey('experiment.id'))

    experiment = relationship("Experiment", back_populates="hyperparams")

def wrap_session(f):
    def g(sql_db, *args, **kwargs):
        if type(sql_db) is Session:
            return f(*args, **kwargs, sql_db=sql_db)
        elif type(sql_db) is Engine:
            engine = sql_db
        else: # type(sql_db) is str:
            engine = create_engine(SQL_PREFIX+sql_db)
        with Session(engine) as session:
            return f(*args, **kwargs, sql_db=session)
    return g

@wrap_session
def get_or_add_row(
        table: Table,
        sql_db: Union[str, Engine, Session],
        **kwargs,
    ) -> Table:
    """
    Tries to find a row in the specified table with data indicated by `kwargs`,
    if not, adds new row. Returns row object.
    """
    criteria = [
        getattr(table, k) == v for k,v in kwargs.items()
        if v is not None
    ]
    query = select(Dataset).where(*criteria)
    result = sql_db.execute(query).first()
    if result is None:
        table_data_str = ' '.join(f"{k}={v}" for k, v in kwargs.items())
        print(f"No {table.__tablename__} found with {table_data_str}, adding row...")
        row = table(**kwargs)
        sql_db.add(row)
        sql_db.commit()
    else:
        row = result[0]
    return row

@wrap_session
def populate_experiment_and_hyperparams(
        args: Namespace,
        sql_db: Union[str, Engine, Session],
        argv: Optional[str] = None
    ) -> Experiment:
    """
    Adds a row to the `Experiment` table containing
    """
    argdict = vars(args).copy()

    # get model id
    model_name = argdict.pop('model')
    peft = argdict.pop('peft_type', None)
    base = argdict.pop('base_model', None)
    model = get_or_add_row(table=Model, sql_db=sql_db, name=model_name, peft=peft, base=base)

    # add experiment row
    experiment = Experiment(
        model_id = model.id,
        argv = argv or sys.argv,
        description = argdict.pop('description', None),
    )
    sql_db.add(experiment)

    # get datasets
    # assume validating if no 'action' arg specified
    # (e.g. with scripts with no training logic)
    if 'dataset' in argdict:
        ds_path = argdict.pop('dataset')
    else:
        ds_path = argdict.pop('input')
    action = argdict.get('action', 'validation')
    
    # script may specify extra train/validation datasets
    train_datasets = argdict.pop('train_datasets', [])
    eval_datasets = argdict.pop('eval_datasets', [])
    if action == 'train':
        train_datasets.append(ds_path)
    # every experiment involves evaluation
    eval_datasets.append(ds_path)
    experiment_datasets = []
    for train_ds in train_datasets:
        ds_name = os.path.basename(train_ds)
        ds = get_or_add_row(table=Dataset, sql_db=sql_db, name=ds_name, split='train')
        experiment_datasets.append(ds)
    for eval_ds in eval_datasets:
        ds_name = os.path.basename(eval_ds)
        ds = get_or_add_row(table=Dataset, sql_db=sql_db, name=ds_name, split='validation')
        experiment_datasets.append(ds)
    experiment.datasets = experiment_datasets

    param_rows = []
    for arg, val in argdict.items():
        row = Hyperparam(param=arg, value=val, experiment_id=experiment.id)
        param_rows.append(row)
    sql_db.add_all(param_rows)
    sql_db.commit()

def init_db(sql_path: str) -> Engine:
    engine = create_engine(SQL_PREFIX+sql_path, echo=True)
    Base.metadata.create_all(engine)
    return engine

if __name__ == '__main__':
    argv = sys.argv[1:]
    sql_path = argv[0]
    init_db(sql_path=sql_path)