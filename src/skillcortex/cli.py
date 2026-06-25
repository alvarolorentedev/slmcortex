from skill_lattice_coder.cli import main as _main


def main(argv: list[str] | None = None) -> int:
    return _main(argv, prog="skillcortex")


if __name__ == "__main__":
    raise SystemExit(main())
