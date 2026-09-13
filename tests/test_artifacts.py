from diagram_restore.artifacts import contact_sheets


def test_contact_sheets_does_not_crash_on_a_small_development_set(tmp_path, tiny_dataset):
    paths = contact_sheets(tiny_dataset, tmp_path / "out")
    for path in paths:
        assert (tmp_path / "out").joinpath(path.split("/")[-1]).exists()


def test_contact_sheets_returns_no_pages_when_there_is_nothing_to_show(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "manifest.jsonl").write_text("")
    paths = contact_sheets(tmp_path / "data", tmp_path / "out")
    assert paths == []
