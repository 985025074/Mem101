import datetime


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def time_to_str(time: datetime.datetime) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
