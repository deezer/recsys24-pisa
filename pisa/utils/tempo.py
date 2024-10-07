import pytz
from datetime import datetime


def high_level_tempo_from_ts(ts, tz='Europe/Paris'):
    dt = datetime.fromtimestamp(ts, tz=pytz.timezone(tz))
    hour_of_day = dt.hour
    day_of_week = dt.weekday() + 1
    return day_of_week, hour_of_day
