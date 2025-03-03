"""Class for storing the results from a MOSAICS run."""

from pydantic import BaseModel, ConfigDict


class MosaicsResult(BaseModel):
    """TODO: docstring."""

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    pass
