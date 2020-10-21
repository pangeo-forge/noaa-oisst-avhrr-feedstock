import fsspec
import datetime
from prefect import Parameter, unmapped
from typing import List
import xarray as xr
from pangeo_forge.pipelines.base import AbstractPipeline
from pangeo_forge.utils import chunked_iterable
from pangeo_forge.tasks.http import download
from prefect import Flow, task
import zarr
import pandas as pd


@task
def chunk(sources, size):
    # TODO: move to pangeo_forge.
    return list(chunked_iterable(sources, size))


@task
def combine_and_write(sources: List[str], target: str, concat_dim: str) -> List[str]:
    """
    Write a batch of intermediate files to a combined zarr store.

    Parameters
    ----------
    sources : List[str]
        A list of URLs pointing to the intermediate files.
    target : str
        The URL for the target combined store.
    concat_dim : str
        The dimension to concatenate along.

    Returns
    -------
    target : str
        The URL of the written combined Zarr store (same as target).k
    """
    double_open_files = [fsspec.open(url).open() for url in sources]
    ds = xr.open_mfdataset(double_open_files, combine="nested", concat_dim=concat_dim)
    # by definition, this should be a contiguous chunk
    ds = ds.chunk({concat_dim: len(sources)})
    mapper = fsspec.get_mapper(target)

    if not len(mapper):
        kwargs = dict(mode="w")
    else:
        kwargs = dict(mode="a", append_dim=concat_dim)
    ds.to_zarr(mapper, **kwargs)
    return target


@task
def source_url(day: str) -> str:
    """
    Format the URL for a specific day.
    """
    day = pd.Timestamp(day)
    source_url_pattern = (
        "https://www.ncei.noaa.gov/data/"
        "sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/"
        "{day:%Y%m}/oisst-avhrr-v02r01.{day:%Y%m%d}.nc"
    )
    return source_url_pattern.format(day=day)


@task
def consolidate_metadata(writes: List[str], target: str) -> None:
    """
    Consolidate the metadata the Zarr group at `target`.

    Parameters
    ----------
    writes : List[str]
        The URLs the combined stores were written to. This is only a
        parameter to introduce a dependency. The actual value isn't used.
    target : str
        The URL for the (combined) Zarr group.
    """
    mapper = fsspec.get_mapper(target)
    zarr.consolidate_metadata(mapper)


class Pipeline(AbstractPipeline):

    # Pipeline constants
    repo = "TomAugspurger/noaa-oisst-avhrr-feedstock"
    name = "noaa-oisst-avhrr-tom"
    concat_dim = "time"
    files_per_chunk = 5

    # Flow parameters
    days = Parameter(
        "days", default=pd.date_range("1981-09-01", "1981-09-10", freq="D").strftime("%Y-%m-%d").tolist()
    )
    variables = Parameter("variables", default=["anom", "err", "ice", "sst"])
    cache_location = Parameter(
        "cache_location", default=f"gs://pangeo-forge-scratch/cache/{name}.zarr"
    )
    target_location = Parameter(
        "target_location", default=f"gs://pangeo-forge-scratch/{name}.zarr"
    )

    @property
    def sources(self):
        # XXX: remote from base class?
        pass

    @property
    def targets(self):
        # XXX: remove from base class?
        return None

    @property
    def flow(self):
        with Flow(self.name) as _flow:
            sources = source_url.map(self.days)
            nc_sources = download.map(
                sources, cache_location=unmapped(self.cache_location)
            )
            chunked = chunk(nc_sources, size=self.files_per_chunk)
            writes = combine_and_write.map(
                chunked, unmapped(self.target_location), unmapped(self.concat_dim)
            )
            consolidate_metadata(writes, self.target_location)

        return _flow
