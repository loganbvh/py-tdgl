import json
import datetime
from typing import Dict

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # scalar complex values only
        if isinstance(obj, (complex, np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        if isinstance(obj, (np.void,)):
            return None

        # float, int, etc.
        if isinstance(obj, (np.generic,)):
            return obj.item()

        if isinstance(obj, datetime.datetime):
            return obj.isoformat()

        return super().default(self, obj)


def json_numpy_obj_hook(d: Dict) -> Dict:
    if set(d.keys()) == {"real", "imag"}:
        return complex(d["real"], d["imag"])

    for key, value in d.items():
        if isinstance(value, str) and len(value) >= 26:
            try:
                d[key] = datetime.datetime.fromisoformat(value)
            except ValueError:
                pass
    return d
