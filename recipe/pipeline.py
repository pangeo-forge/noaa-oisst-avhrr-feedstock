import os
from datetime import timedelta
from typing import List

import dask
import fsspec
import xarray as xr
from numcodecs import Blosc
from pangeo_forge.pipelines.base import AbstractPipeline
from pangeo_forge.tasks.http import download
from pangeo_forge.utils import chunked_iterable
from prefect import Flow, task, unmapped
from prefect.environments import DaskKubernetesEnvironment
from prefect.environments.storage import Docker

# options
name = "noaa-oisst-avhrr"
cache_location = f"gs://pangeo-forge-scratch/cache/{name}.zarr"
target_location = f"gs://pangeo-forge-scratch/{name}.zarr"


days = pd.date_range("1981-09-01", "1981-09-10", freq="D")
variables = ['anom', 'err', 'ice', 'sst']


def get_encoding(ds):
    compressor = Blosc()
    encoding = {key: {"compressor": compressor} for key in ds.data_vars}
    return encoding


@task(max_retries=1, retry_delay=timedelta(seconds=1))
def download(source_url: str, cache_location: str) -> str:
    """
    Download a remote file to a cache.
    Parameters
    ----------
    source_url : str
        Path or url to the source file.
    cache_location : str
        Path or url to the target location for the source file.
    Returns
    -------
    target_url : str
        Path or url in the form of `{cache_location}/hash({source_url})`.
    """
    fs = fsspec.get_filesystem_class(cache_location.split(':')[0])(token='cloud')

    target_url = os.path.join(cache_location, str(hash(source_url)))

    # there is probably a better way to do caching!
    try:
        fs.open(target_url)
        return target_url
    except FileNotFoundError:
        pass

    with fsspec.open(source_url, mode="rb") as source:
        with fs.open(target_url, mode="wb") as target:
            target.write(source.read())
    return target_url


@task
def combine_and_write(sources, target, append_dim, first=True):
    # while debugging this, I had itermittent fsspec / hdf5 read errors related to
    # "trying to read from a closed file"
    # but they seem to have gone away for now
    double_open_files = [fsspec.open(url).open() for url in sources]
    ds = xr.open_mfdataset(double_open_files, combine="nested", concat_dim=concat_dim)
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


class OISSTPipeline(AbstractPipeline):

    # what properies should be class variables
    concat_dim = "time"
    files_per_chunk = 5

    # vs runtime parameters?
    def __init__(self, cache_location, target_location, variables, days):
        self.name = name
        self.days = days
        # these two feel like they should be set by the runtime
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
            source_url_pattern.format(yyyymm=key.strftime("%Y%m"), yyyymmdd=key.strftime("%Y%m%d"))
            for day in self.days
        ]

        return source_urls


    @property
    def flow(self):

        with Flow(self.name, storage=self.storage, environment=self.environment) as _flow:
            # download to cache
            nc_sources = download.map(
                self.sources,
                cache_location=unmapped(self.cache_location),
            )

            first = True
            write_tasks = []
            for source_group in chunked_iterable(nc_sources, self.files_per_chunk):
                write_task = combine_and_write(source_group, self.target_location, self.concat_dim, first=first)
                write_tasks.append(write_task)
                first = False
            cm = consolidate_metadata(target_path)

        return _flow


    @property
    def targets(self):
        return [self.target_location]


    # Very confusing to have this in the middle of the recipe!
    # Feels totally out of place.
    @property
    def environment(self):
        environment = DaskKubernetesEnvironment(
            min_workers=1, max_workers=30,
            scheduler_spec_file="recipe/job.yaml",
            worker_spec_file="recipe/worker_pod.yaml",
        )
        return environment

    @property
    def storage(self):
        storage = Docker(
            "pangeoforge",
            dockerfile="recipe/Dockerfile",
            prefect_directory="/home/jovyan/prefect",
            python_dependencies=[
                "git+https://github.com/pangeo-forge/pangeo-forge@master",
                "prefect==0.13.6",
            ],
            image_tag="latest",
        )
        return storage


# why are we calling this in the recipe file?
pipeline = TerraclimatePipeline(cache_location, target_location, variables, years)
# feel like this should happen in the runtime, not in the recipe definiton


if __name__ == "__main__":
    pipeline.flow.validate()

    print(pipeline.flow)
    print(pipeline.flow.environment)
    print(pipeline.flow.parameters)
    print(pipeline.flow.sorted_tasks())
    print("Registering Flow")
    pipeline.flow.register(project_name="pangeo-forge")
