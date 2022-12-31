# WebColor dataset

A dataset for web page coloring task consisting of e-commerce mobile web pages,
adapted from the [Klarna Product Page Dataset](https://github.com/klarna/product-page-dataset). (todo: put figure)

## Download dataset

```bash
./download.sh {main,cache,text,image}
```

-   "main" (1.5 GB)

    -   Main dataset
    -   Split file
    -   Note: Processing to DGL format takes about 50 minutes.

-   "cache" (693 MB, recommended)

    -   Processed DGL cache
    -   Split file

-   "text" (356 MB, optional)

    -   Text data for reference

-   "image" (6.3 GB, optional)

    -   Image data for reference

## Data format

The dataset is organized in HDF5 format, and its contents can be accessed using
the [h5py library](https://docs.h5py.org/en/stable/quick.html) as follows:

```python
import h5py

f = h5py.File('data/webcolor_v1.1.hdf5')
for data_name, grp in f.items():
    for node_id, dset in grp.items():
        node_id = int(node_id)
        parent_id = int(dset[()])
        node_attrs = dict(dset.attrs)
```

Variable|Type|Description
---|---|---
`data_name`|str|Name to identify the web page in the original datasets
`node_id`|int|Node ID
`parent_id`|int|Parent node ID (`-1` indicates no parent, *i.e.*, root)
`node_attrs`|dict|Metadata corresponding to the node (element)

Node attrs

Key|Type|Description
---|---|---
"text_color"|str|Text color (computed value of the CSS property `color`, text elements only)
"background_color"|str|Background color (computed value of the CSS property `background-color`)
"sibling_order"|int|Order between sibling elements
"html_tag"|str|HTML tag
"text_feat"|numpy.ndarray<br>(dtype=float32,<br>shape=(13,))|Low-level features of text (text elements only)
"img_feat"|numpy.ndarray<br>(dtype=float32,<br>shape=(13,))|Low-level features of image (image elements only)
"bgimg_feat"|numpy.ndarray<br>(dtype=float32,<br>shape=(12,))|Low-level features of background image (only for elements with a background image)

Details of the text and image features can be found in the [supplemental material](https://arxiv.org/abs/2212.11541).

## Data split

```
Train: 27,630 pages
Val: 3,190 pages
Test: 13,228 pages
```

## License

This dataset is based on the [Klarna Product Page Dataset](https://github.com/klarna/product-page-dataset) and is licensed under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/).

## Versions

1.1: fix bug on text color (Dec 29, 2022)

1.0: v1 release (Dec 23, 2022)
