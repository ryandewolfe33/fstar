import importlib.metadata
import fstar # Required to see if package loads


def test_package_metadata() -> None:
    # Verifies the package metadata and version were packaged properly
    version = importlib.metadata.version("fstar-compare")
    assert version is not None
    assert version != "0.0.0"


if __name__ == "__main__":
    test_package_metadata()
    print("Smoke tests passed successfully!")