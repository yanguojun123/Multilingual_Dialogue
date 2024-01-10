from zhdate import ZhDate
from datetime import datetime
for i in range(1900, 2100):
    if ZhDate.from_datetime(datetime(i, 11, 28)).lunar_month == 10 and ZhDate.from_datetime(datetime(i, 11, 28)).lunar_day == 16:
        print(i)


print(type(ZhDate.from_datetime(datetime(2022, 3, 27)).lunar_month))