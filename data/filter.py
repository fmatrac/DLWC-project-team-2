import csv
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed/usa_mentions_csv")


def iter_daily_files(raw_dir: Path = RAW_DIR):
    yield from sorted(raw_dir.glob("raw_gdelt_*.csv"))


def concerns_usa(row):
    actor1_cc = row[6].strip() if len(row) > 6 else ""
    actor2_cc = row[16].strip() if len(row) > 16 else ""
    actor1_geo_cc = row[37].strip() if len(row) > 37 else ""
    actor2_geo_cc = row[47].strip() if len(row) > 47 else ""
    action_geo_cc = row[57].strip() if len(row) > 57 else ""

    return (
        actor1_cc == "USA"
        or actor2_cc == "USA"
        or actor1_geo_cc == "US"
        or actor2_geo_cc == "US"
        or action_geo_cc == "US"
    )


def output_name_for(path: Path) -> Path:
    name = path.stem
    if name.startswith("raw_"):
        name = name[len("raw_"):]
    return OUTPUT_DIR / f"{name}_usa_only.csv"


def filter_all_files(raw_dir: Path = RAW_DIR, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    file_count = 0

    for path in iter_daily_files(raw_dir):
        file_count += 1
        kept = 0
        seen = 0
        output_file = output_name_for(path)

        with open(path, "r", encoding="utf-8", newline="") as in_f, open(output_file, "w", encoding="utf-8", newline="") as out_f:
            reader = csv.reader(in_f, delimiter="\t")
            writer = csv.writer(out_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for row in reader:
                if not row:
                    continue
                seen += 1
                total_in += 1
                if concerns_usa(row):
                    writer.writerow(row)
                    kept += 1
                    total_out += 1

        print(f"[file] {path.name}: kept {kept}/{seen} -> {output_file}")

    print(f"[done] processed {file_count} files | kept {total_out}/{total_in} rows -> {output_dir}")


if __name__ == "__main__":
    filter_all_files()