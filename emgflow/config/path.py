from __future__ import annotations

import os


_DEFAULT_PATHS = {
    "NinaproDB2": "",
    "NinaproDB4": "",
    "NinaproDB6": "",
    "NinaproDB7": "",
}

_ENV_VARS = {
    "NinaproDB2": "NINAPRO_DB2_ROOT",
    "NinaproDB4": "NINAPRO_DB4_ROOT",
    "NinaproDB6": "NINAPRO_DB6_ROOT",
    "NinaproDB7": "NINAPRO_DB7_ROOT",
}


def get_dataset_root(dataset_name: str) -> str:
    try:
        env_name = _ENV_VARS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc
    return os.environ.get(env_name, _DEFAULT_PATHS[dataset_name])


PATH_NinaproDB2 = get_dataset_root("NinaproDB2")
PATH_NinaproDB4 = get_dataset_root("NinaproDB4")
PATH_NinaproDB6 = get_dataset_root("NinaproDB6")
PATH_NinaproDB7 = get_dataset_root("NinaproDB7")
