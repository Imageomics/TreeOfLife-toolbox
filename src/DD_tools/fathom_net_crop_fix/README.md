tool to fix fathom net cropped images (crop again from original with updated algorithm)

Additional config fields:

* `uuid_table_path` - path to uuid table to filter out
* `look_up_table_path` - path to look up table with `uuid - file_name` information
* `filtered_by_size` - original csv that was used for cropping (contains `uuid` matches and crop coordinates)
* `data_transfer_table` - csv that contains match between ToL dataset file and original image