This is a general instruction on how to download TreeOfLife200M dataset using `distributed-downloader` package and other
tools.

There are four types of download you are going to need to run to get the dataset (all of them can be run in parallel):

- [`general/fast`](General_download_README.md) - is the fast and distributed download of the most images from the GBIF
  source
- [`safe`](Safe_download_README.md) - is slower version of `general` download, retreiving images from heavy rate-limited
  servers from GBIF and EoL sources.
- [`bioscan`](BIOSCAN_download_README.md) - is special download for the bioscan dataset
- [`fathomNet`](FathomNet_download_README.md) - is special download for the fathomNet dataset
