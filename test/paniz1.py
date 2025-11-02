# -*- coding: utf-8 -*-

# --- 1) تو رفتگی (Indentation) و ساختار شرطی ---
if 5 > 2:
    print("five is greater than two!")

# --- 2) متغیرها و رشته‌ها ---
x = 5
y = "Hello, World!"

# --- 3) کامنت‌ها (یک‌خطی و چندخطی) ---
# this is a comment.
print("Hello, World!")

# چند خطی با سه‌گانه کوتیشن (اگر لازم باشه)
"""
this is a comment.
written in
more than just one line
"""
print("hello, world!")

# --- 4) نمایش متغیرها ---
x = 5
y = "john"
print(x)
print(y)

# --- 5) تبدیل نوع (casting) ---
x = str(3)   # x will be '3' (string)
y = int(3)   # y will be 3   (int)
z = float(3) # z will be 3.0 (float)
print(x, type(x))
print(y, type(y))
print(z, type(z))

# --- 6) type() مثال ---
x = 5
y = "john"
print(type(x))
print(type(y))

# --- 7) حساس به حروف بزرگ/کوچک (case sensitivity) ---
a = 4
A = "sally"
print(a)  # 4
print(A)  # "sally"

# --- 8) اختصاص چند مقدار در یک خط ---
x, y, z = "Orange", "Banana", "Cherry"
print(x)
print(y)
print(z)

# یک مقدار برای چندین متغیر
x = y = z = "Orange"
print(x)
print(y)
print(z)

# Unpacking از لیست
fruits = ["apple", "banana", "cherry"]
x, y, z = fruits
print(x)
print(y)
print(z)

# --- 9) خروجی رشته‌ها ---
x = "python is awesome"
print(x)

x = "Python"
y = "is"
z = "awesome"
print(x, y, z)        # با کاما بینشان فاصله می‌گذارد
print(x + " " + y + " " + z)  # با + و اضافه کردن فاصله‌ها هم می‌شود

# --- 10) جمع اعداد ---
x = 5
y = 10
print(x + y)  # 15

# --- 11) تلاش برای جمع کردن عدد و رشته (خطا) ---
x = 5
y = 'john'
# print(x + y)  # این دستور خطا می‌دهد؛ عدد و رشته را نمی‌توان مستقیم جمع کرد
# راه درست:
print(str(x) + y)  # تبدیل عدد به رشته و سپس جمع
print(x, y)        # یا استفاده از کاما در print

# --- 12) متغیر سراسری و توابع ---
# مثال 1: خواندن متغیر سراسری داخل تابع
x = "awesome"
def myfunc():
    print("python is " + x)

myfunc()

# مثال 2: تعریف متغیر محلی با همان نام (ن overwrite نمیکند متغیر سراسری)
x = "awesome"
def myfunc2():
    x = "fantastic"   # این x محلی است
    print("python is " + x)

myfunc2()
print("python is " + x)  # هنوز مقدار سراسری دست نخورده

# مثال 3: تغییر متغیر سراسری داخل تابع با استفاده از global
x = "awesome"
def myfunc3():
    global x
    x = "fantastic"

print("قبل از myfunc3:", x)
myfunc3()
print("بعد از myfunc3:", x)

# تعیین نوع داده

x= 7
print(type(x))

# تنطیم نوع داده

x = "Hello World"
# display x:
print(x)
# display the data type of x:
print(type(x))

x = 20
#display x:
print(x)
#display the data type of x:
print(type(x))

x = ["apple", "banana", "cherry"]
# display x:
print(x)
# display the data type of x:
print(type(x))

x = frozenset({"apple", "banana", "cherry"})
#display x:
print(x)
#display the data type of x:
print(type(x))

# تنظیم نوع داده خاص

x = str("Hello World")
# display x:
print(x)
# display the data type of x:
print(type(x))

x = int(20)
#display x:
print(x)
#display the data type of x:
print(type(x))

x = list(("apple", "banana", "cherry"))
#display x:
print(x)
#display the data type of x:
print(type(x))

x = bool(5)
#display x:
print(x)
#display the data type of x:
print(type(x))

# اعداد در پایتون

x = 1
y = 2.8
z = 1j
print(type(x))
print(type(y))
print(type(z))

# casting

# convert from int to float:
x = float(1)
# convert from float to int:
y = int(2.8)
# convert from int to complex:
z = complex(1)
print(x)
print(y)
print(z)
print(type(x))
print(type(y))
print(type(z))

# رشته ها در پایتون

print("Hello")
print('Hello')
print("He is called 'Johnny' ")
print('He is called "Johnny" ')

# اختصاص رشته به یک متفییر

a = "hello"
print(a)

#رشته چند خطی

a = """Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua."""
print(a)

# رشته آرایه

a = "Hello, World!"
print(a[1])

#پیمایش رشته با استفاده از حلقه

for x in "banana":
    print(x)

# طول رشته

a = "Hello, World!"
print(len(a))

# برسی رشته

txt = "The best things in life are free!"
print("free" in txt)

# Slicing

b = "Hello, World!"
print(b[2:5])
b = "Hello, World!"
print(b[:5])
b = "Hello, World!"
print(b[2:])
b = "Hello, World!"
print(b[-5:-2])

# متد های پایتون

a = "Hello, World!"
print(a.upper())
a = "Hello, World!"
print(a.lower())
a = " Hello, World! "
print(a.strip())
a = "Hello, World!"
print(a.replace("H", "J"))
a = "Hello, World!"
b = a.split(",")
print(b)

# String Concatenation

a = "Hello"
b = "World"
c = a + b
print(c)
a = "Hello"
b = "World"
c = a + " " + b
print(c)

# پایتون- قالب بندی-رشته ها

age = 36
txt = f"My name is John, I am {age}"
print(txt)

#فرمت رشته از طریق f

price = 59
txt = f"The price is {price} dollars"
print(txt)
price = 59
txt = f"The price is {price:.2f} dollars"
print(txt)
txt = f"The price is {20 * 59} dollars"
print(txt)
name = "Alice"
age = 30
greeting = f"Hello, I'm {name} and I'm {age} years old."
print(greeting)
x = 10
y = 20
result = f"The sum of {x} and {y} is {x + y}."
print(result)

# انواع بکس اسلش ها

txt = 'It\'s alright.'
print(txt)
txt = "This will insert one \\ (backslash)."
print(txt)
txt = "Hello\nWorld!"
print(txt)
txt = "Hello\tWorld!"
print(txt)
txt = "Hello \bWorld!"
print(txt)
txt = "Hello\rWorld!"
print(txt)

# بولین توی پایتون

a = 200
b = 33
if b > a:
    print("b is greater than a")
else:
    print("b is not greater than a")








