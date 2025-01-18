# This code is part of Tergite
#
# (C) Copyright Martin Ahindura, 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from typing import Dict, Any, List, Type

from sqlalchemy import ColumnExpressionArgument
from sqlmodel import SQLModel, Session, select, create_engine

from app.libs.store.base import Store, Filter


class SqlStore(Store):
    """The store based on SQL relational database"""

    def __init__(self, db_uri: str):
        self._engine = create_engine(db_uri)
        SQLModel.metadata.create_all(self._engine)

    def insert(self, items: List[SQLModel], model: Type[SQLModel]=None) -> List[SQLModel]:
        with Session(self._engine) as session:
            session.add_all(items)
            session.commit()
            session.refresh(items)
        return items

    def find(self, filters: Filter, model: Type[SQLModel]=None) -> List[SQLModel]:
        with Session(self._engine) as session:
            whereclauses = _to_whereclauses(model=model, filters=filters)
            statement = select(model).where(*whereclauses)
            results = session.exec(statement)
        return results

    def update(self, filters: Filter, update: Dict[str, Any], model: Type[SQLModel]=None) -> List[SQLModel]:
        pass

    def delete(self, filters: Filter, model: Type[SQLModel]=None) -> List[SQLModel]:
        pass


def _to_whereclauses(model: Type[SQLModel], filters: Filter) ->  List[ColumnExpressionArgument]:
    """Converts filters to output expected by sqlalchemy's whereclauses

    Args:
        model: the model construct the filters against
        filters: the dataclass of things to match against

    Returns:
        the SQLModel-specific filters, also called "whereclauses"
    """
    raise NotImplementedError("implement to_where_clause")
