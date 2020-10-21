import os
from datetime import timedelta

import fsspec
import xarray as xr
from numcodecs import Blosc
from pangeo_forge.pipelines.base import AbstractPipeline
from pangeo_forge.utils import chunked_iterable
from pangeo_forge.tasks.http import download
from prefect import Flow, task
import zarr

# # options
# cache_location = f"gs://pangeo-forge-scratch/cache/{name}.zarr"
# target_location = f"gs://pangeo-forge-scratch/{name}.zarr"


# days = pd.date_range("1981-09-01", "1981-09-10", freq="D")
# variables = ['anom', 'err', 'ice', 'sst']



@task
def combine_and_write(sources, target, append_dim, first=True):
    double_open_files = [fsspec.open(url).open() for url in sources]
    ds = xr.open_mfdataset(double_open_files, combine="nested", concat_dim=append_dim)
    # by definition, this should be a contiguous chunk
    ds = ds.chunk({append_dim: len(sources)})

    if first:
        kwargs = dict(mode="w")
    else:
        kwargs = dict(mode="a", append_dim=append_dim)
    mapper = fsspec.get_mapper(target)
    ds.to_zarr(mapper, **kwargs)


@task
def consolidate_metadata(target):
    mapper = fsspec.get_mapper(target)
    zarr.consolidate_metadata(mapper)


class Pipeline(AbstractPipeline):

    concat_dim = "time"
    files_per_chunk = 5
    repo = "pangeo-forge/noaa-oisst-avhrr-feedstock/"
    name = "noaa-oisst-avhrr"

    def __init__(self, cache_location, target_location, variables, days):
        self.days = days
        self.cache_location = cache_location
        self.target_location = target_location

    @property
    def sources(self):
        source_url_pattern = (
            "https://www.ncei.noaa.gov/data/"
            "sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/"
            "{yyyymm}/oisst-avhrr-v02r01.{yyyymmdd}.nc"
        )
        source_urls = [
            source_url_pattern.format(yyyymm=day.strftime("%Y%m"), yyyymmdd=day.strftime("%Y%m%d"))
            for day in self.days
        ]

        return source_urls

    @property
    def flow(self):

        with Flow(self.name) as _flow:
            nc_sources = [
                download(x, cache_location=self.cache_location)
                for x in self.sources
            ]

            first = True
            write_tasks = []
            for source_group in chunked_iterable(nc_sources, self.files_per_chunk):
                write_task = combine_and_write(source_group, self.target_location, self.concat_dim, first=first)
                write_tasks.append(write_task)
                first = False
            consolidate_metadata(self.target_location)

        return _flow


    @property
    def targets(self):
        return [self.target_location]
