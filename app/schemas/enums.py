from enum import StrEnum


class MissionStatus(StrEnum):
    READY_FOR_APPROVAL = "READY_FOR_APPROVAL"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    REJECTED = "REJECTED"


class LegType(StrEnum):
    TRANSIT = "TRANSIT"
    SEARCH_PATTERN = "SEARCH_PATTERN"
    LOITER = "LOITER"
    SENSOR_TASK = "SENSOR_TASK"
    RETURN_TO_BASE = "RETURN_TO_BASE"


class SensorMode(StrEnum):
    EO = "EO"
    IR = "IR"
    OFF = "OFF"


class PatternName(StrEnum):
    LAWNMOWER = "lawnmower"
    EXPANDING_SQUARE = "expanding_square"
    SECTOR = "sector"
