#باز کردن یک فایل روی سرور
f = open("demofile.txt")
print(f.read())
#باز کردن فایل با ادرس مشخص
f = open("D:\\myfiles\\welcome.txt")
print(f.read())
#استقاده از دستور wirh
with open("demofile.txt") as f:
    print(f.read())
#خواندن خط ب خط و بستن فایل
f = open("demofile.txt")
print(f.readline())
f.close()
#خواندن تمام خطوط با حلقه for
f = open("demofile.txt")
for line in f:
    print(line.strip())
f.close()
#خواندن بخش های خاصی از فایل 
#خواندن پنج کارکتر اول
with open("demofile.txt") as f:
    print(f.read(5))
#خواندن دو خط باreadline 
with open("demofile.txt") as f:
    print(f.readline())
    print(f.readline())
#اجرای حلقه روی فایل
with open("demofile.txt") as f:
    for x in f:
        print(x)
#نوشتن و اضافه کردن به فایل
with open("demofile.txt", "a") as f:
    f.write("Now the file has more content!")

# باز کردن و خواندن برای مشاهده تغییرات
with open("demofile.txt") as f:
    print(f.read())
#بازنویسی محتوای فایل
with open("demofile.txt", "w") as f:
    f.write("Woops! I have deleted the content!")
# باز کردن و خواندن برای مشاهده نتیجه
with open("demofile.txt") as f:
    print(f.read())
#ایجاد یک فایل جدید
# ایجاد یک فایل جدید به نام myfile.txt
f = open("myfile.txt", "x")
#حذف یک فایل
import os
os.remove("demofile.txt")
#بررسی وجود فایل قبل از حذف
import os
if os.path.exists("demofile.txt"):
    os.remove("demofile.txt")
else:
    print("The file does not exist")
#حذف پوشه
import os
os.rmdir("myfolder")
#نمایش تاریخ و زمان فعلی
import datetime
x = datetime.datetime.now()
print(x)
#متدstrftime()
from datetime import datetime
now = datetime.now()

# نمایش کامل تاریخ و زمان
print(now.strftime("%Y-%m-%d %H:%M:%S")) 

# نمایش فقط نام ماه و سال
print(now.strftime("%B %Y"))

# نمایش روز هفته و ساعت (AM/PM)
print(now.strftime("%A %I:%M %p"))
