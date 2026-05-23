from enum import Enum


class MissionStatus(str, Enum):
    READY_FOR_APPROVAL = "READY_FOR_APPROVAL"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    REJECTED = "REJECTED"


class LegType(str, Enum):
    TRANSIT = "TRANSIT"
    SEARCH_PATTERN = "SEARCH_PATTERN"
    LOITER = "LOITER"
    SENSOR_TASK = "SENSOR_TASK"
    RETURN_TO_BASE = "RETURN_TO_BASE"


class SensorMode(str, Enum):
    EO = "EO"
    IR = "IR"
    OFF = "OFF"


class PatternName(str, Enum):
    LAWNMOWER = "lawnmower"
    EXPANDING_SQUARE = "expanding_square"
    SECTOR = "sector"
