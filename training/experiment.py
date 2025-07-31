import yaml
from pathlib import Path
import copy

class Experiment:
    def __init__(self, config_path: str, schema_path: str = Path(__file__).resolve().parents[0] / "schema.yaml"):
        self.config_path = Path(config_path)
        self.schema_path = Path(schema_path)
        self.schema = None
        self.config = None

        self._load_schema()
        self._load_config()
        self._apply_defaults()
        self._validate_config()

    def _load_schema(self):
        with open(self.schema_path, "r") as f:
            self.schema = yaml.safe_load(f)

    def _load_config(self):
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f) or {}

    def _apply_defaults(self):
        """Recursively merge schema defaults into config."""
        def merge(schema_section, config_section):
            merged = {}
            for key, val in schema_section.items():
                # Nested section
                if isinstance(val, dict) and "default" not in val:
                    merged[key] = merge(
                        val,
                        config_section.get(key, {}) if isinstance(config_section.get(key), dict) else {}
                    )
                else:
                    # Use config override if available, else schema default
                    merged[key] = config_section.get(key, val["default"])
            return merged

        self.config = merge(self.schema, self.config)

    def _validate_config(self):
        """Check if options are respected."""
        def validate(config_section, schema_section, path=""):
            for key, val in schema_section.items():
                if isinstance(val, dict) and "default" not in val:
                    validate(config_section[key], val, path + key + ".")
                else:
                    if "options" in val:
                        if config_section[key] not in val["options"]:
                            raise ValueError(
                                f"{path+key} must be one of {val['options']}, got {config_section[key]}"
                            )
        validate(self.config, self.schema)

    def print_config_with_descriptions(self):
        """Print config values alongside descriptions."""
        def recurse(config_section, schema_section, indent=0):
            for key, val in schema_section.items():
                if isinstance(val, dict) and "default" not in val:
                    print(" " * indent + f"{key}:")
                    recurse(config_section[key], val, indent + 2)
                else:
                    desc = val.get("description", "")
                    print(" " * indent + f"{key}: {config_section[key]} — {desc}")

        recurse(self.config, self.schema)

    def run(self):
        print("Final experiment configuration:")
        self.print_config_with_descriptions()
        print("\n[Running experiment logic here...]")
        # TODO: Replace with actual training/evaluation pipeline


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parents[0] / "config.yaml"

    exp = Experiment(config_path)
    exp.run()
