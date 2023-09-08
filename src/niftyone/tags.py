
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Union

import fiftyone as fo
import pandas as pd

TAGS = [
    "Fail",
    "Borderline",
    "Head motion",
    "Eye spillover",
    "Non-eye spillover",
    "Coil failure",
    "Global noise",
    "Local noise",
    "EM interference",
    "Bad FoV",
    "Wrap-around",
    "Aliasing ghosts",
    "Other ghosts",
    "Intensity non-uniformity",
    "Temporal field variation",
    "Postproc artifact",
    "Other artifact",
]

_TAG_SET = set(TAGS)


class GroupTags:
    """
    QC tags for individual data acquisitions. Interfaces between FiftyOne datasets and
    pandas DataFrames, CSV, JSON.
    """
    def __init__(self, tags_dict: Dict[str, Any]):
        self.tags_dict = tags_dict

    @classmethod
    def from_dataset(cls, dataset: fo.Dataset):
        """
        Extract tags from a FiftyOne dataset.
        """

        def new_record() -> Dict[str, Any]:
            record = {tag: False for tag in TAGS}
            record["_Extra"] = []
            return record

        tags_dict = defaultdict(new_record)
        for sample in dataset:
            label: fo.Classification = sample["group_key"]
            record = tags_dict[label.label]

            for tag in label.tags:
                if tag in _TAG_SET:
                    record[tag] = True
                else:
                    record["_Extra"].append(tag)
            tags_dict[label.label] = record
        return cls(dict(tags_dict))

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        return cls(df.to_dict(orient="index"))

    @classmethod
    def from_csv(cls, path: Union[str, Path]):
        # TODO: is there a better way to read the _Extra column?
        df = pd.read_csv(path, index_col=0, converters={"_Extra": ast.literal_eval})
        return cls.from_df(df)

    @classmethod
    def from_json(cls, path: Union[str, Path]):
        with open(path, "r") as f:
            return cls(json.load(f))

    def apply(self, dataset: fo.Dataset):
        """
        Apply tags to a given FiftyOne dataset. This will overwrite any existing tags.
        """
        for sample in dataset:
            label: fo.Classification = sample["group_key"]
            label.tags = []
            if label.label in self.tags_dict:
                for tag, value in self.tags_dict[label.label].items():
                    if tag == "_Extra":
                        for extra_tag in value:
                            label.tags.append(extra_tag)
                    elif value:
                        label.tags.append(tag)

    def to_df(self):
        return pd.DataFrame.from_dict(self.tags_dict, orient="index")

    def to_csv(self, path: Union[str, Path]):
        self.to_df().to_csv(path)

    def to_json(self, path: Union[str, Path]):
        with open(path, "w") as f:
            json.dump(self.tags_dict, f)

    def equals(self, other: "GroupTags"):
        return self.tags_dict == other.tags_dict
