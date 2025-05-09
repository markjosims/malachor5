from sqlalchemy import create_engine, Column, Integer, Float, ForeignKey, DateTime, String, Engine, UnicodeText
from sqlalchemy.orm import relationship, declarative_base, Mapped, mapped_column
from typing import *
import sys

"""
Classes for managing SQL database for storing experiment results.

Adapted from Gemini output on May 8th 2025.
Original Gemini prompt:
```
I'm uploading an image of a SQL database schema. Write me a file with SQL alchemy classes for each table in this schema.
```
Schema stored at: https://drawsql.app/teams/ucsd-6/diagrams/ml-experiments
"""

Base = declarative_base()

class Model(Base):
    __tablename__ = 'model'
    id: Mapped[int] = mapped_column(primary_key=True)
    base: Mapped[int] = mapped_column(Integer, ForeignKey("model.id"))
    name: Mapped[String] = mapped_column(String(20), unique=True)
    path: Mapped[String] = mapped_column(String(255))
    experiments: Mapped[List["Experiment"]] = relationship(back_populates="model")
    peft = Column(String(20))

class Dataset(Base):
    __tablename__ = 'dataset'
    id: Mapped[int] = mapped_column(primary_key=True)
    language = Column(String(20))
    name: Mapped[str] = mapped_column(String(20), unique=True)  
    path: Mapped[str] = mapped_column(String(255), unique=True)

class Experiment(Base):
    __tablename__ = 'experiment'
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey('model.id'))
    train_data: Mapped[int] = mapped_column(ForeignKey('dataset.id'))
    eval_data: Mapped[int] = mapped_column(ForeignKey('dataset.id'))
    test_data: Mapped[int] = mapped_column(ForeignKey('dataset.id'))
    argv = Column(String(255))
    name: Mapped[String] = mapped_column(String(20), unique=True)

    model = relationship("Model", back_populates="experiments")  # Added relationship
    train_dataset = relationship("Dataset", foreign_keys=[train_data])
    eval_dataset = relationship("Dataset", foreign_keys=[eval_data])
    test_dataset = relationship("Dataset", foreign_keys=[test_data])
    train_events: Mapped[List["TrainEvent"]] = relationship(back_populates="experiment")
    eval_events: Mapped[List["EvalEvent"]] = relationship(back_populates="experiment")
    test_events: Mapped[List["TestEvent"]] = relationship(back_populates="experiment")
    hyperparams: Mapped[List["Hyperparam"]] = relationship(back_populates="experiment")

class TrainEvent(Base):
    __tablename__ = 'train_event'
    id: Mapped[int] = mapped_column(primary_key=True)
    tag = Column(String(20))
    value = Column(String(20))
    experiment_id: Mapped[int] = mapped_column(ForeignKey('experiment.id'))

    experiment: Mapped["Experiment"] = relationship(back_populates="train_events")

class EvalEvent(Base):
    __tablename__ = 'eval_event'
    id: Mapped[int] = mapped_column(primary_key=True)
    tag = Column(String(20))
    value = Column(Float)
    experiment_id: Mapped[int] = mapped_column(ForeignKey('experiment.id'))

    experiment: Mapped["Experiment"] = relationship(back_populates="eval_events")
    eval_params: Mapped[List["EvalParam"]] = relationship(back_populates="eval_event")

class TestEvent(Base):
    __tablename__ = 'test_event'
    id: Mapped[int] = mapped_column(primary_key=True)
    tag = Column(String(20))
    value = Column(Float)
    experiment_id: Mapped[int] = mapped_column(ForeignKey('experiment.id'))

    experiment: Mapped["Experiment"] = relationship(back_populates="test_events")

class EvalParam(Base):
    __tablename__ = 'eval_param'
    id: Mapped[int] = mapped_column(primary_key=True)
    param = Column(String(20))
    value = Column(UnicodeText())
    eval_event_id: Mapped[int] = mapped_column(ForeignKey('eval_event.id'))

    eval_event: Mapped["EvalEvent"] = relationship(back_populates="eval_params")

class EventParam(Base):
    __tablename__ = 'event_param'
    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey('train_event.id'))
    param_id: Mapped[int] = mapped_column(ForeignKey('eval_param.id'))

class Hyperparam(Base):
    __tablename__ = 'hyperparam'
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey('experiment.id'))
    param = Column(String(20))
    value = Column(String(20))

    experiment: Mapped["Experiment"] = relationship(back_populates="hyperparams")
    

def init_db(sql_path: str) -> Engine:
    engine = create_engine("sqlite:///"+sql_path, echo=True)
    Base.metadata.create_all(engine)
    return engine


if __name__ == '__main__':
    argv = sys.argv[1:]
    sql_path = argv[0]
    init_db(sql_path=sql_path)