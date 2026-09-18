import os
import pytest
import icepack


try:
    earthdata_username = os.environ["EARTHDATA_USERNAME"]
    earthdata_password = os.environ["EARTHDATA_PASSWORD"]
    earthdata_auth = True
except KeyError:
    earthdata_auth = False


reason = "Need EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables"


@pytest.mark.skipif(not earthdata_auth, reason=reason)
def test_fetching_data():
    function_names = [
        "measures_antarctica",
        "measures_greenland",
        "bedmachine_antarctica",
        "bedmachine_greenland",
        "mosaic_of_antarctica",
        "mosaic_of_greenland",
        "randolph_glacier_inventory",
    ]
    for function_name in function_names:
        print(function_name)
        fn = getattr(icepack.datasets, "fetch_" + function_name)
        filenames = fn()
        assert filenames
