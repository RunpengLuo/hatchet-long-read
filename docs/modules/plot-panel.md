# `plot-panel`
`plot-panel` performs multi-sample copy-number panel plotting, composed from several per-sample CN solutions.

## Input

A panel TSV (`--panel_file`) listing the per-sample BBC UCN solution paths, plus the reference `--genome_size` / `--region_bed`. See [compute-cn Output](compute-cn.md#output) for the `results/` UCN file layout.

| Parameter | Default | Description |
|---|---|---|
| `--panel_file` | *(required)* | Panel TSV listing per-sample BBC UCN paths |
| `--genome_size` | *(required)* | Reference chromosome sizes file |
| `--region_bed` | *(required)* | Reference chromosome BED file |
| `-o` / `--out_file` | *(required)* | Output figure path (e.g., `panel.svg`) |
| `--dpi` | 300 | Image resolution |
| `--transparent` | False | Transparent background |
| `--title` | `panel` | Plot title |

## Usage

```console
$ hatchet plot-panel --help
usage: hatchet plot-panel [-h] --panel_file PANEL_FILE
                          --genome_size GENOME_SIZE --region_bed REGION_BED
                          [--width WIDTH] [--height HEIGHT]
                          [--show_clone_name | --no-show_clone_name]
                          [--show_prop | --no-show_prop]
                          [--show_ploidy | --no-show_ploidy]
                          [--min_prop MIN_PROP] [--dpi DPI] [--transparent]
                          [--title TITLE] -o OUT_FILE [--plot_1d2d]
                          [--plot_summary]
```

## Main parameters

### Panel input

`--panel_file` is a TSV listing the per-sample BBC UCN solution paths to compose into one multi-sample panel; `-o`/`--out_file` sets the output figure (e.g. `panel.svg`).

### Layout and annotations

- **Layout (`--width`, `--height`, `--dpi`, `--transparent`, `--title`).** Overall panel width and per-row height in inches (defaults 20 and 1), output resolution, background, and title.

- **Annotations (`--show_clone_name`, `--show_prop`, `--show_ploidy`, `--min_prop`).** Toggle clone names, proportions, and per-clone ploidy on each CN profile; `--min_prop` (default 0.01) hides clones below that proportion.

### Extras

`--plot_1d2d` also runs `plot-cn` per panel row (needs a `PATH_TO_BBC` column in the panel file); `--plot_summary` emits per-sample purity and ploidy barplots.

Panel display parameters:

| Parameter | Default | Description |
|---|---|---|
| `--width` | 20 | Panel image width |
| `--height` | 1 | Panel image height per row |
| `--show_clone_name` | False | Draw clone names on the CN profile |
| `--show_prop` | False | Draw clone proportions on the CN profile |
| `--show_ploidy` | False | Draw per-clone ploidy on the CN profile |
| `--min_prop` | 0.01 | Hide tumor clones below this proportion from the panel |
| `--plot_1d2d` | False | Also run `plot-cn` per panel row (requires a `PATH_TO_BBC` column) |
| `--plot_summary` | False | Emit per-sample purity + ploidy barplots (one page per metric per `cancer_type`) |

## Output

The composed multi-sample panel figure written to `--out_file`. With `--plot_1d2d` it also emits per-row `plot-cn` figures, and with `--plot_summary` per-sample purity and ploidy barplots.
