# subtitle-translator

`subtitle-translator` is a Linux CLI utility that inspects an MKV file, lets the
user select an existing subtitle track, translates that track with an OpenAI
model, and muxes the translated subtitles back into the MKV as an additional
track.

## Requirements

- Python 3.12+
- MKVToolNix installed and available in `PATH`
  - `mkvmerge`
  - `mkvextract`
- `OPENAI_API_KEY` set in the environment

## Install

Install MKVToolNix first. This is required; the program will not run without
`mkvmerge` and `mkvextract`.

Fedora:

```bash
sudo dnf install -y mkvtoolnix
```

Debian/Ubuntu:

```bash
sudo apt install -y mkvtoolnix
```

Then create the Python virtual environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

You also need an OpenAI API key in the environment before running the tool:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

```bash
subtitle-translator movie.mkv
```

Optional flags:

```bash
subtitle-translator movie.mkv --track-id 2 --target-lang fr --model gpt-5.4
subtitle-translator movie.mkv --output movie.fr.translated.mkv
subtitle-translator movie.mkv --in-place
```

The default interaction is:

1. Inspect the MKV subtitle tracks
2. Choose a source subtitle track
3. Choose the target language
4. Choose whether to write a new file or replace the original after success

The translator sends the whole subtitle file in a single request whenever the
model accepts it. If the payload is too large, it falls back to splitting the
subtitle file into the smallest number of large contiguous chunks needed to
complete the translation.
