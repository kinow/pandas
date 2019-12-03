import os

import pytest

import pandas as pd
import pandas.compat as compat
import pandas.util.testing as tm


def test_rands():
    r = tm.rands(10)
    assert len(r) == 10


def test_rands_array_1d():
    arr = tm.rands_array(5, size=10)
    assert arr.shape == (10,)
    assert len(arr[0]) == 5


def test_rands_array_2d():
    arr = tm.rands_array(7, size=(10, 10))
    assert arr.shape == (10, 10)
    assert len(arr[1, 1]) == 7


def test_numpy_err_state_is_default():
    expected = {"over": "warn", "divide": "warn", "invalid": "warn", "under": "ignore"}
    import numpy as np

    # The error state should be unchanged after that import.
    assert np.geterr() == expected


def test_convert_rows_list_to_csv_str():
    rows_list = ["aaa", "bbb", "ccc"]
    ret = tm.convert_rows_list_to_csv_str(rows_list)

    if compat.is_platform_windows():
        expected = "aaa\r\nbbb\r\nccc\r\n"
    else:
        expected = "aaa\nbbb\nccc\n"

    assert ret == expected


def test_create_temp_directory():
    with tm.ensure_clean_dir() as path:
        assert os.path.exists(path)
        assert os.path.isdir(path)
    assert not os.path.exists(path)


@pytest.mark.parametrize("strict_data_files", [True, False])
def test_datapath_missing(datapath):
    with pytest.raises(ValueError, match="Could not find file"):
        datapath("not_a_file")


def test_datapath(datapath):
    args = ("data", "iris.csv")

    result = datapath(*args)
    expected = os.path.join(os.path.dirname(os.path.dirname(__file__)), *args)

    assert result == expected


def test_rng_context():
    import numpy as np

    expected0 = 1.764052345967664
    expected1 = 1.6243453636632417

    with tm.RNGContext(0):
        with tm.RNGContext(1):
            assert np.random.randn() == expected1
        assert np.random.randn() == expected0


def test_assert_almost_equal():
    # see https://github.com/pandas-dev/pandas/issues/25068
    df1 = pd.DataFrame([
        0.00016,
        -0.154526,
        -0.20580199999999998])
    df2 = pd.DataFrame([
        0.00015981824253685772,
        -0.15452557802200317,
        -0.20580188930034637])
    pd.testing.assert_frame_equal(
        df1, df2, check_exact=False, check_less_precise=3)
    df1 = pd.DataFrame([
        0.15,
        0.099999])
    df2 = pd.DataFrame([
        0.16,
        0.01])
    pd.testing.assert_frame_equal(
        df1, df2, check_exact=False, check_less_precise=1)
