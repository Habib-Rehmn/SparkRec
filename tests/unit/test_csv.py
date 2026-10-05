import pytest

from sparkrec.ingestion.csv_records import check_header, parse_partition


def test_parser_preserves_quotes_and_rejects_wrong_width():
    rows = list(
        parse_partition(
            0,
            iter(
                [
                    "movieId,title,genres",
                    '1,"A ""Quoted"", Title",Drama',
                    "2,Title,Comedy,",
                    '3,"Unterminated,Comedy',
                    "4,Missing genres",
                ]
            ),
            3,
        )
    )
    assert rows[0][:3] == ("1", 'A "Quoted", Title', "Drama")
    assert rows[0][-1] is None
    assert all(row[-1] is not None for row in rows[1:])


def test_only_first_partition_skips_header():
    assert list(parse_partition(1, iter(["1,2,3"]), 3))[0][:3] == ("1", "2", "3")


def test_header_must_match_order(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text("movieId,userId,rating,timestamp\n")
    with pytest.raises(ValueError, match="expected header"):
        check_header(path, ("userId", "movieId", "rating", "timestamp"))
