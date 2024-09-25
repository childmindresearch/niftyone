from pathlib import Path

import pytest
from bids2table import BIDSEntities, BIDSTable

from niftyone.metrics import create_niftyone_metrics_tsv


@pytest.mark.b2t()
class TestCreateNiftyOneMetricsTSV:
    def test_create_tsv_no_metrics(
        self,
        b2t_index: BIDSTable,
        tmp_path: Path,
    ):
        out_dir = tmp_path / "output"
        qc_dir = tmp_path / "qc"
        record = b2t_index.filter_multi(sub="01", suffix="T1w").nested.iloc[0]
        entities = BIDSEntities.from_dict(record["ent"])

        out_path = entities.with_update(
            ext=".tsv", extra_entities={"metrics": "QCMetrics"}
        ).to_path(prefix=out_dir)
        assert not out_path.exists()

        create_niftyone_metrics_tsv(
            record=record, entities=entities, out_dir=out_dir, qc_dir=qc_dir
        )

    def test_create_tsv_metrics(
        self,
        b2t_index: BIDSTable,
        tmp_path: Path,
        qc_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        out_dir = tmp_path / "output"
        record = b2t_index.filter_multi(sub="01", suffix="T1w").nested.iloc[0]
        entities = BIDSEntities.from_dict(record["ent"])

        out_path = entities.with_update(
            ext=".tsv", extra_entities={"metrics": "QCMetrics"}
        ).to_path(prefix=out_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        create_niftyone_metrics_tsv(
            record=record, entities=entities, out_dir=out_dir, qc_dir=qc_dir
        )

        assert "Creating" in caplog.text
        assert out_path.exists()

    def test_out_path_exists(self, b2t_index: BIDSTable, tmp_path: Path, qc_dir: Path):
        out_dir = tmp_path / "output"
        record = b2t_index.filter_multi(sub="01", suffix="T1w").nested.iloc[0]
        entities = BIDSEntities.from_dict(record["ent"])

        out_path = entities.with_update(
            ext=".tsv", extra_entities={"metrics": "QCMetrics"}
        ).to_path(prefix=out_dir)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        out_path.touch()
        assert out_path.exists()

        create_niftyone_metrics_tsv(
            record=record, entities=entities, out_dir=out_dir, qc_dir=qc_dir
        )
