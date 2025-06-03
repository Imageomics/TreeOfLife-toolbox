import marimo

__generated_with = "0.12.8"
app = marimo.App(width="full")


@app.cell
def _():
    import collections
    import dataclasses
    import pathlib
    import re

    import beartype
    import marimo as mo
    return beartype, collections, dataclasses, mo, pathlib, re


@app.cell
def _(mo):
    mo.md(
        r"""
        # Experiment #1

        This document helps readers develop intuition around how PDQ (a perceptual hash algorithm developed by Meta) groups images by similarity.

        ## Takeaways

        * **PDQ is an effective perceptual hash algorithm.**
        * **Distances of 10 or less are genuinely identical** across 70K randomly sampled images from TreeOfLife-10M.
        * I could not run on BIOSCAN-5M due to unknown memory constraints. Further investigation is needed.

        ## Experimental Procedure

        1. For each algorithm (PDQ, pHash) and each dataset (iNat2021/train, ToL-10M/train, ToL-2, BioSCAN-5M/train), hash all of the images in the dataset.
        2. Within each algorithm/dataset pair, sample 30K pairs of hashes and record their pairwise distances.
        3. Plot a histogram of these distances.
        4. For each threshold in [0, 2, 5, 10, 20, 50, 100, 200, 256], randomly sample 10 images, then get all the images within the dataset that match those "query" images.
        5. Record these images to develop human intuition for what different thresholds and algorithms lead to.

        This document contains several sample outputs.
        Full outputs are avaiable on disk, and steps to reproduce these outputs are at the bottom of this document.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## PDQ

        * PDQ assigns a 256 bit hash to every image.
        * Distance is measured as total number of different bits.
        * The minimum distance is 0 (identical) and the maximum distance is 256 (all bits different)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(f"""
    ### iNat2021 Training Split Distance Distribution
    {mo.image(src="plots/inat21_train-pdq.png")}

    Note the log scale along the y-axis.

    We randomly sampled 30K images in the iNat2021 training split. Then we calculated all the distances between each pair of images (30K x (30K + 1) / 2 = roughly 450K pairs).
    Then we plotted a histogram of the distances.

    The spike at 0 is the 30K images that are identical to themselves.
    """)
    return


@app.cell
def _(mo):
    mo.md(f"""
    ### TreeOfLife-10M Training Split Distance Distribution
    {mo.image(src="plots/tol-10m_train-pdq.png")}
    """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### BIOSCAN-5M Training Split Distance Distribution

        *Missing because of memory errors while calculating hashes.*
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Image Clusters

        We want to develop intuition around what different distances mean within a given dataset.
        For example, what does a distance of 2 look like between two images?
        What does a distance of 10 look like?
        And so on.
        """
    )
    return


@app.cell
def _(max_dist_input, mo):
    mo.hstack([mo.md("Maximum distance:"), max_dist_input], justify="start")
    return


@app.cell
def _(max_dist_input, show_clusters):
    show_clusters("tol-10m/train", max_dist=max_dist_input.value, interactive=False)
    return


@app.cell
def _(max_dist_input, show_clusters):
    show_clusters("inat21/train", max_dist=max_dist_input.value, interactive=False)
    return


@app.cell
def _(mo):
    max_dist_input = mo.ui.number(start=0, stop=256, step=1, value=50)
    return (max_dist_input,)


@app.cell
def _(beartype, dataclasses, mo, pathlib, re):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class MatchInfo:
        """Holds information about a single match image."""

        match_index: int
        distance: int
        match_path: str


    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class QueryData:
        """Holds information about a query image and its matches."""

        query_index: int
        query_path: str
        matches: list[MatchInfo] = dataclasses.field(
            default_factory=list
        )  # Initialize with empty list

        def get_min_distance(self) -> int | None:
            """Returns the minimum distance among matches, or None if no matches."""
            if not self.matches:
                return None
            # Assumes matches list might not be sorted yet, finds minimum
            return min(match.distance for match in self.matches)

        def sort_matches_by_distance(self):
            """Sorts the internal matches list by distance."""
            self.matches.sort(key=lambda m: m.distance)


    def show_clusters(name: str, *, max_dist: int, interactive: bool = True):
        """
        Scans a directory structure for image clusters and displays them in Marimo.

        Args:
            name: The base name for the cluster directory (e.g., "tol-10m_train"). Slashes will be replaced with underscores for the path.
            interactive: If True, uses an accordion for display. Otherwise, displays sequentially.

        Returns:
            A Marimo vstack object containing the visual representation of clusters.
        """
        # Construct the base path for cluster data
        # Replaces '/' with '_' to match potential dataset naming conventions in paths
        base_path = pathlib.Path("clusters", name.replace("/", "_"))

        # Initialize list to hold Marimo output elements
        output_elements = [mo.md(f"# Image Clusters for '{name}'"), mo.md(f"*(distance less than {max_dist})*")]


        # Regex to extract match index and distance from filename
        match_pattern = re.compile(r"match-(\d+)_dist-(\d+)\.jpg")

        # --- 1. Data Collection ---
        # Check if the base directory exists
        if not base_path.is_dir():
            output_elements.append(
                mo.md(f"**Error:** Base directory '{base_path}' not found.")
            )
            return mo.vstack(output_elements)

        # Dictionary to store QueryData objects, keyed by query_index
        # {query_index: QueryData_instance}
        all_query_data: dict[int, QueryData] = {}

        # Iterate through distance range directories (e.g., '0_2', '5_10')
        for range_dir in base_path.iterdir():
            if not range_dir.is_dir():
                continue

            # Iterate through query index directories (e.g., '1058616', '1225592')
            for query_dir in range_dir.iterdir():
                if not query_dir.is_dir():
                    continue

                try:
                    query_index = int(query_dir.name)
                except ValueError:
                    continue  # Skip if directory name isn't an integer

                query_path = query_dir / f"query-{query_index}.jpg"

                # Ensure the query image exists
                if not query_path.is_file():
                    continue

                # Get or create the QueryData object for this index
                if query_index not in all_query_data:
                    all_query_data[query_index] = QueryData(
                        query_index=query_index, query_path=str(query_path)
                    )
                # If query already seen, ensure path is consistent (optional check)
                # elif all_query_data[query_index].query_path != str(query_path):
                #     print(f"Warning: Inconsistent path for query {query_index}")


                # Find all match files within this specific query directory
                for item in query_dir.iterdir():
                    if not item.is_file():
                        continue

                    match = match_pattern.match(item.name)
                    if not match:
                        continue

                    # --- Apply Distance Filter ---
                    distance = int(match.group(2))
                    if distance > max_dist:
                        continue # Skip this match if distance is too high

                    # Create MatchInfo object
                    match_info = MatchInfo(
                        match_index=int(match.group(1)),
                        distance=distance,
                        match_path=str(item),  # Store path as string
                    )
                    # Append valid match to the QueryData object
                    all_query_data[query_index].matches.append(match_info)


        # Filter out queries that ended up with no matches *after* distance filtering
        queries_with_matches = [
            qd for qd in all_query_data.values() if qd.matches
        ]

        # Check if any matches remain after filtering
        if not queries_with_matches:
            output_elements.append(
                mo.md(
                    f"**Info:** No matches found with distance <= {max_dist} in the specified directory structure."
                )
            )
            # Optionally report how many queries were found initially
            if all_query_data:
                 output_elements.append(mo.md(f"Found {len(all_query_data)} query images initially."))

            return mo.vstack(output_elements)

        # --- 2. Sort Matches Per Query ---
        # --- 3. Determine Query Order based on Minimum Distance ---
        query_order_data = []
        for query_data in queries_with_matches:
             # Sort matches for this query by distance (ascending)
            query_data.sort_matches_by_distance() # Use the method on the dataclass

            # Get the minimum distance (first item after sorting)
            min_distance = query_data.matches[0].distance # Access attribute directly

            # Append (min_distance, query_index) for overall sorting
            query_order_data.append((min_distance, query_data.query_index))

        # --- 4. Sort Query Order ---
        # Sorts primarily by min_distance, then by query_index as a tie-breaker
        query_order_data.sort()

        # --- 5. Generate Marimo Output ---
        output_elements.append(
            mo.md(f"Displaying queries sorted by their closest match distance (up to {max_dist}).")
        )

        accordion_items = {}
        for min_dist, query_index in query_order_data:
            # Retrieve the corresponding QueryData object
            query_data = all_query_data[query_index]

            # Create elements for the horizontal row (query | match1 | match2 ...)
            row_elements = []

            # Add Query Image + Label VStack
            row_elements.append(
                 mo.image(
                    src=query_data.query_path, # Use attribute from dataclass
                    width=128,
                    caption=f"Query {query_data.query_index}" # Use attribute
                )
            )

            # Add Match Images + Label VStacks (using the sorted matches list)
            for match_info in query_data.matches: # Iterate through sorted list in dataclass
                row_elements.append(
                     mo.image(
                        src=match_info.match_path, # Use attribute
                        width=128,
                        caption=f"Match: {match_info.match_index} (Dist: {match_info.distance})", # Use attributes
                    )
                )

            # Combine the query and its matches horizontally
            row = mo.hstack(row_elements, justify="start", gap=1, wrap=True)

            # Add to accordion or directly to output based on 'interactive' flag
            if interactive:
                accordion_items[f"Query {query_index} (Min Dist: {min_dist})"] = row
            else:
                # Add separator if not using accordion
                if len(output_elements) > 1: # Avoid separator before the first item
                     output_elements.append(mo.md("---"))
                output_elements.append(row)


        # If using accordion, add it now
        if interactive and accordion_items:
            output_elements.append(
                mo.accordion(accordion_items, multiple=True, lazy=True)
            )

        # Combine all generated elements vertically
        return mo.vstack(output_elements)
    return MatchInfo, QueryData, show_clusters


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Reproduce These Results

        1. Hash the images in the datasets.
        2. Calculate image distributions to get a feel for it.
        3. Save image clusters.
        4. Run this notebook (replace $BASE_DIR with your data root and YOUR_ACCOUNT with your account ID).

        ### Hash Images

        iNat21:

        ```sh
        DATA=$BASE_DIR/foundation_model/inat21/raw uv run python main.py hash-all \
          --algorithm pdq \
          --dataset inat21/train \
          --data-root $DATA \
          --n-workers 60 \
          --batch-size 512 \
          --slurm-acct YOUR_ACCOUNT
        ```

        TreeOfLife-10M:

        ```sh
        DATA=$BASE_DIR/open_clip/data/evobio10m-v3.3/224x224 uv run python experiments.py hash-all \
            --dataset tol-10m/train \
            --algorithm pdq \
            --data-root $DATA \
            --n-workers 60 \
            --batch-size 512 \
            --slurm-acct YOUR_ACCOUNT
        ```

        Then look in `./hashes`.

        ### Calculate Image Distributions

        iNat21:

        ```sh
        DATA=$BASE_DIR/foundation_model/inat21/raw uv run python experiments.py make-hist \
            --algorithm pdq \
            --dataset inat21/train \
            --data-root $DATA \
            --hashes hashes/inat21_train-pdq.npy \
            --n 70_000
        ```

        TreeOfLife-10M:

        ```sh
        DATA=$BASE_DIR/open_clip/data/evobio10m-v3.3/224x224 uv run python experiments.py make-hist \
            --algorithm pdq \
            --hashes hashes/tol-10m_train-pdq.npy \
            --dataset tol-10m/train \
            --data-root $DATA \
            --n 70_000
        ```

        Then look in `./plots`.

        ### Save Image Clusters

        iNat21:

        ```sh
        DATA=$BASE_DIR/foundation_model/inat21/raw uv run experiments.py make-clusters \
            --algorithm pdq \
            --hashes hashes/inat21_train-pdq.npy \
            --dataset inat21/train \
            --data-root $DATA \
            --n 70_000 \
            --seed 18
        ```

        TreeOfLife-10M:

        ```sh
        DATA=$BASE_DIR/open_clip/data/evobio10m-v3.3/224x224 uv run experiments.py make-clusters \
            --algorithm pdq \
            --hashes hashes/tol-10m_train-pdq.npy \
            --dataset tol-10m/train \
            --data-root $DATA \
            --n 70_000 \
            --seed 18
        ```

        Then look in `./clusters`.

        ### Run Notebook

        ```sh
        uv run marimo edit experiments/exp1.py
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
