"""Unit tests for mole_jepa.data.datacomp."""

import pathlib
import unittest.mock

import pytest
import torch

from mole_jepa.data.datacomp import build_datacomp_loader, find_tar_shards, split_shards

# ---------------------------------------------------------------------------
# TestFindTarShards
# ---------------------------------------------------------------------------


class TestFindTarShards:
    def test_finds_flat_tars(self, tmp_path: pathlib.Path) -> None:
        for name in ("a.tar", "b.tar", "c.tar"):
            (tmp_path / name).touch()
        shards = find_tar_shards(tmp_path)
        assert len(shards) == 3
        assert all(s.endswith(".tar") for s in shards)

    def test_returns_sorted_paths(self, tmp_path: pathlib.Path) -> None:
        for name in ("z.tar", "a.tar", "m.tar"):
            (tmp_path / name).touch()
        shards = find_tar_shards(tmp_path)
        assert shards == sorted(shards)

    def test_finds_nested_tars(self, tmp_path: pathlib.Path) -> None:
        # Mirrors the chunk_NNNN layout produced by prepare_datacomp_main.py.
        for chunk in ("chunk_0000", "chunk_0001"):
            (tmp_path / chunk).mkdir()
            (tmp_path / chunk / "00000.tar").touch()
        shards = find_tar_shards(tmp_path)
        assert len(shards) == 2

    def test_ignores_non_tar_files(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "data.tar").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "data.parquet").touch()
        shards = find_tar_shards(tmp_path)
        assert len(shards) == 1

    def test_raises_when_no_tars_found(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError, match="No .tar shards"):
            find_tar_shards(tmp_path)

    def test_returns_absolute_string_paths(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "shard.tar").touch()
        shards = find_tar_shards(tmp_path)
        assert all(pathlib.Path(s).is_absolute() for s in shards)

    def test_accepts_string_path(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "shard.tar").touch()
        shards = find_tar_shards(str(tmp_path))
        assert len(shards) == 1


# ---------------------------------------------------------------------------
# TestSplitShards
# ---------------------------------------------------------------------------


class TestSplitShards:
    def _paths(self, n: int) -> list[str]:
        return [f"/shards/{i:05d}.tar" for i in range(n)]

    def test_zero_fraction_returns_all_for_train(self) -> None:
        paths = self._paths(10)
        train, val = split_shards(paths, val_fraction=0.0)
        assert train == paths
        assert val == []

    def test_nonzero_fraction_reserves_tail(self) -> None:
        paths = self._paths(20)
        train, val = split_shards(paths, val_fraction=0.1)
        # 10% of 20 = 2 val shards
        assert len(val) == 2
        assert len(train) == 18
        # val comes from the tail
        assert val == paths[-2:]
        assert train == paths[:-2]

    def test_at_least_one_val_shard_for_tiny_dataset(self) -> None:
        # Even with val_fraction of 0.01, we should get at least 1 val shard.
        paths = self._paths(5)
        _, val = split_shards(paths, val_fraction=0.01)
        assert len(val) >= 1

    def test_train_and_val_are_disjoint(self) -> None:
        paths = self._paths(100)
        train, val = split_shards(paths, val_fraction=0.15)
        assert set(train).isdisjoint(set(val))

    def test_train_plus_val_covers_all(self) -> None:
        paths = self._paths(100)
        train, val = split_shards(paths, val_fraction=0.2)
        assert sorted(train + val) == sorted(paths)

    def test_train_is_stable_prefix(self) -> None:
        """Extending the shard list should never change the train prefix."""
        paths = self._paths(50)
        train_50, _ = split_shards(paths, val_fraction=0.1)
        # Adding 50 more shards to the end should not touch the first 45
        # training shards from the 50-shard run.
        train_100, _ = split_shards(paths + self._paths(50), val_fraction=0.1)
        assert train_50 == train_100[: len(train_50)]


# ---------------------------------------------------------------------------
# TestBuildDatacompLoader
# ---------------------------------------------------------------------------


class TestBuildDatacompLoader:
    """Tests for build_datacomp_loader.

    We mock out WebDataset and DataLoader so the tests run without real tar
    files or a network connection.  The assertions focus on which kwargs are
    forwarded to the DataLoader constructor, not on actual data iteration.
    """

    @pytest.fixture
    def data_config(self) -> object:
        from mole_jepa.config import DataConfig

        return DataConfig(
            batch_size=4,
            num_workers=0,
            image_field="jpg",
            caption_field="txt",
        )

    def _patch_pipeline(self) -> unittest.mock.MagicMock:
        """Return a MagicMock that simulates the wds pipeline chain."""
        pipeline = unittest.mock.MagicMock()
        # Each chained call returns the same mock so .decode().shuffle().map()
        # all resolve to the same object, which is then accepted by DataLoader.
        pipeline.decode.return_value = pipeline
        pipeline.shuffle.return_value = pipeline
        pipeline.map.return_value = pipeline
        return pipeline

    def _call_with_mocks(
        self,
        data_config: object,
        *,
        shuffle: bool = False,
        extra_dl_patch: bool = False,
    ) -> tuple[unittest.mock.MagicMock, unittest.mock.MagicMock, object]:
        """Run build_datacomp_loader with all heavy deps mocked out.

        Returns ``(mock_wds, mock_dl, loader)``.
        """
        pipeline = self._patch_pipeline()
        mock_dl_instance = unittest.mock.MagicMock(spec=torch.utils.data.DataLoader)

        with (
            unittest.mock.patch(
                "mole_jepa.data.datacomp.transforms.build_image_transform",
                return_value=unittest.mock.MagicMock(),
            ),
            unittest.mock.patch(
                "mole_jepa.data.datacomp.transforms.build_tokenizer",
                return_value=unittest.mock.MagicMock(),
            ),
            unittest.mock.patch(
                "mole_jepa.data.datacomp.wds.WebDataset", return_value=pipeline
            ) as mock_wds,
            unittest.mock.patch(
                "mole_jepa.data.datacomp.torch.utils.data.DataLoader",
                return_value=mock_dl_instance,
            ) as mock_dl,
        ):
            loader = build_datacomp_loader(
                ["/fake/shard.tar"],
                data_config,  # type: ignore[arg-type]
                torch.device("cpu"),
                shuffle=shuffle,
            )
        return mock_wds, mock_dl, loader

    def test_returns_dataloader(self, data_config: object) -> None:
        _, mock_dl, loader = self._call_with_mocks(data_config)
        assert loader is mock_dl.return_value

    def test_batch_size_propagates(self, data_config: object) -> None:
        _, mock_dl, _ = self._call_with_mocks(data_config)
        _, kwargs = mock_dl.call_args
        assert kwargs["batch_size"] == data_config.batch_size  # type: ignore[union-attr]

    def test_no_pin_memory_for_cpu(self, data_config: object) -> None:
        _, mock_dl, _ = self._call_with_mocks(data_config)
        _, kwargs = mock_dl.call_args
        assert kwargs["pin_memory"] is False

    def test_shuffle_false_passes_shardshuffle_false(self, data_config: object) -> None:
        mock_wds, _, _ = self._call_with_mocks(data_config, shuffle=False)
        _, kwargs = mock_wds.call_args
        assert kwargs.get("shardshuffle") is False

    def test_shuffle_true_passes_shardshuffle_true(self, data_config: object) -> None:
        mock_wds, _, _ = self._call_with_mocks(data_config, shuffle=True)
        _, kwargs = mock_wds.call_args
        assert kwargs.get("shardshuffle") is True

    def test_no_persistent_workers(self, data_config: object) -> None:
        _, mock_dl, _ = self._call_with_mocks(data_config)
        _, kwargs = mock_dl.call_args
        assert kwargs["persistent_workers"] is False

    def test_preprocess_fn_handles_bytes_image_and_caption(
        self, data_config: object
    ) -> None:
        """The _preprocess closure must decode bytes images and captions."""
        import io

        import torch
        from PIL import Image

        # Encode a tiny 4×4 RGB image to JPEG bytes.
        buf = io.BytesIO()
        Image.new("RGB", (4, 4), color=(128, 64, 32)).save(buf, format="JPEG")
        jpg_bytes = buf.getvalue()

        captured_fn: list = []
        pipeline = self._patch_pipeline()

        # Intercept the callable passed to pipeline.map().
        def _capture_map(fn):  # type: ignore[no-untyped-def]
            captured_fn.append(fn)
            return pipeline

        pipeline.map.side_effect = _capture_map

        # The image transform just returns a fixed tensor; tokenizer returns ids/mask.
        img_tensor = torch.zeros(3, 4, 4)
        ids = torch.zeros(64, dtype=torch.long)
        mask = torch.ones(64, dtype=torch.long)

        with (
            unittest.mock.patch(
                "mole_jepa.data.datacomp.transforms.build_image_transform",
                return_value=lambda _img: img_tensor,
            ),
            unittest.mock.patch(
                "mole_jepa.data.datacomp.transforms.build_tokenizer",
                return_value=lambda _cap: (ids, mask),
            ),
            unittest.mock.patch(
                "mole_jepa.data.datacomp.wds.WebDataset", return_value=pipeline
            ),
            unittest.mock.patch(
                "mole_jepa.data.datacomp.torch.utils.data.DataLoader",
                return_value=unittest.mock.MagicMock(),
            ),
        ):
            build_datacomp_loader(
                ["/fake/shard.tar"],
                data_config,  # type: ignore[arg-type]
                torch.device("cpu"),
                shuffle=False,
            )

        assert len(captured_fn) == 1
        preprocess = captured_fn[0]

        # Test with bytes image + bytes caption.
        pv, out_ids, out_mask = preprocess(
            {"jpg": jpg_bytes, "txt": b"a caption in bytes"}
        )
        assert torch.equal(pv, img_tensor)
        assert torch.equal(out_ids, ids)
        assert torch.equal(out_mask, mask)

    def test_preprocess_fn_handles_pil_image_and_str_caption(
        self, data_config: object
    ) -> None:
        """The _preprocess closure must also accept a PIL image and str caption."""
        import torch
        from PIL import Image

        captured_fn: list = []
        pipeline = self._patch_pipeline()

        def _capture_map(fn):  # type: ignore[no-untyped-def]
            captured_fn.append(fn)
            return pipeline

        pipeline.map.side_effect = _capture_map

        img_tensor = torch.zeros(3, 4, 4)
        ids = torch.zeros(64, dtype=torch.long)
        mask = torch.ones(64, dtype=torch.long)

        with (
            unittest.mock.patch(
                "mole_jepa.data.datacomp.transforms.build_image_transform",
                return_value=lambda _img: img_tensor,
            ),
            unittest.mock.patch(
                "mole_jepa.data.datacomp.transforms.build_tokenizer",
                return_value=lambda _cap: (ids, mask),
            ),
            unittest.mock.patch(
                "mole_jepa.data.datacomp.wds.WebDataset", return_value=pipeline
            ),
            unittest.mock.patch(
                "mole_jepa.data.datacomp.torch.utils.data.DataLoader",
                return_value=unittest.mock.MagicMock(),
            ),
        ):
            build_datacomp_loader(
                ["/fake/shard.tar"],
                data_config,  # type: ignore[arg-type]
                torch.device("cpu"),
                shuffle=False,
            )

        preprocess = captured_fn[0]
        pil_img = Image.new("RGB", (4, 4))
        pv, out_ids, out_mask = preprocess({"jpg": pil_img, "txt": "a string caption"})
        assert torch.equal(pv, img_tensor)
        assert torch.equal(out_ids, ids)
