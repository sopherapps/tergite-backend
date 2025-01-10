# This code is part of Tergite
#
# (C) Martin Ahindura (2025)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import copy

from qiskit.qobj import QobjExperimentHeader


def copy_header_with(header: QobjExperimentHeader, **kwargs):
    """Copies a new header from the old header with new kwargs set

    Args:
        header: the original QobjExperimentHeader header
        kwargs: the extra key-word args to set on the header

    Returns:
        a copy QobjExperimentHeader instance
    """
    new_header = copy.deepcopy(header)
    for k, v in kwargs.items():
        setattr(new_header, k, v)

    return new_header
