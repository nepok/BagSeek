import argparse
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator, Optional

from dotenv import load_dotenv
from mcap.reader import make_reader

PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
if PARENT_ENV.exists():
    load_dotenv(dotenv_path=PARENT_ENV)


def resolve_mcap_files(path: Path) -> list[Path]:
    """Return a list of .mcap files to read from the given path."""
    if path.is_file() and path.suffix == ".mcap":
        return [path]

    if path.is_dir():
        files = sorted(path.glob("*.mcap"))
        if files:
            return files
        raise FileNotFoundError(f"No .mcap files found in directory: {path}")

    raise FileNotFoundError(f"Path does not point to an .mcap file or directory: {path}")


def iter_messages(mcap_files: Iterable[Path], topics: Optional[list[str]] = None) -> Iterator[tuple]:
    """Yield (source_file, schema, channel, message) tuples for the given Mcap files."""
    for file_path in mcap_files:
        with file_path.open("rb") as handle:
            reader = make_reader(handle)
            for schema, channel, message in reader.iter_messages(topics=topics):
                yield file_path, schema, channel, message


def list_topics(mcap_files: Iterable[Path]) -> OrderedDict[str, str]:
    """Return an ordered mapping of topic -> schema name for the provided files."""
    topics: OrderedDict[str, str] = OrderedDict()

    for file_path in mcap_files:
        with file_path.open("rb") as handle:
            reader = make_reader(handle)
            seen_channels: set[int] = set()
            for schema, channel, _message in reader.iter_messages():
                if channel.id in seen_channels:
                    continue
                seen_channels.add(channel.id)
                topics.setdefault(channel.topic, schema.name if schema else "<unknown>")

    return topics


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick MCAP reader to inspect topics and messages.")
    parser.add_argument(
        "path",
        nargs="?",
        default="/home/nepomuk/sflnas/DataReadOnly334/tractor_data/autorecord/rosbag2_2025_07_23-12_58_03/rosbag2_2025_07_23-12_58_03_5.mcap",
        help="Path to an .mcap file or a directory containing .mcap files (defaults to ROSBAGS_DIR).",
    )
    parser.add_argument(
        "--topic",
        "-t",
        action="append",
        dest="topics",
        help="Filter to one or more topics (can be repeated).",
    )
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="Only list available topics and exit.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Maximum number of messages to print.",
    )

    args = parser.parse_args()

    if not args.path:
        parser.error("No path provided. Pass a path explicitly or set ROSBAGS_DIR in the environment.")

    try:
        files = resolve_mcap_files(Path(args.path))
    except FileNotFoundError as exc:
        parser.error(str(exc))
        return

    if args.list_topics:
        for topic, schema_name in list_topics(files).items():
            print(f"{topic}: {schema_name}")
        return

    count = 0
    topics = args.topics if args.topics else None

    for file_path, schema, channel, message in iter_messages(files, topics=topics):
        count += 1
        if isinstance(message.data, (bytes, bytearray)):
            head = message.data[:16]
            data_preview = head.hex() + ("…" if len(message.data) > len(head) else "")
        else:
            data_preview = message.data
        print(
            f"{file_path.name}:{channel.topic} ({schema.name if schema else '<unknown>'}) "
            f"log_time={message.log_time} data={data_preview}"
        )

        if args.limit is not None and count >= args.limit:
            break


if __name__ == "__main__":
    main()
