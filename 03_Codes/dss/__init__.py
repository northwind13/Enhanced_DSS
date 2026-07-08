"""DisasterAware Decision Support System (rebuilt, phase 1).

Phase 1a implements the observation side of the framework's architecture:
the domain is partitioned into Local DSS regions (one agent per region,
plus one Global DSS coordinator), each agent observes its own cells and
extracts the ten bounded observation features. Sensors are
emulated in this phase: the observation reads the live simulation state
directly; the sensor catalogue arrives in a later phase.
"""

from .regions import Region, partition, partition_n
from .features import ten_features, FEATURE_ORDER
from .sensors import (Sensor, SensorNetwork, SENSOR_CATALOG, CHANNELS,
                      LAMBDA_CONF)

__all__ = ["Region", "partition", "partition_n", "ten_features",
           "FEATURE_ORDER", "Sensor", "SensorNetwork", "SENSOR_CATALOG",
           "CHANNELS", "LAMBDA_CONF"]

# bumped on every dss change; checked by the app freshness gate
DSS_BUILD = 2
