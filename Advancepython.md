<h1>Python Learning</h1>

[TOC]

# 1.类的静态属性、类方法、静态方法

**静态属性 == 数据属性**

若想方便使用想将类的函数属性同对象的数据属性一样可供对象直接调用，可以在类中的函数前加上装饰器@property，这样就将函数属性
转换为类似数据属性一样可供直接调用（封装）*看起来像调用数据属性一样*
但是不可被修该（不同于数据属性），静态属性可以访问类的数据属性和实例的数据属性

```python
class cal:
    cal_name = '计算器'
    def __init__(self,x,y):
        self.x = x
        self.y = y

    @property           #在cal_add函数前加上@property，使得该函数可直接调用，封装起来
    def cal_add(self):
        return self.x + self.y

c = cal(10,11)
print(c.cal_name)   #>>> '计算器'   调用类的数据属性
print(c.cal_add)    #>>> 21         这样调用类函数属性看起来跟调用数据属性一样  c.cal_name >>> '计算器'
c.cal_add = 10      #这样修改会报错，因为不可被修改p
```



如果不想通过实例来调用类的函数属性，而**直接用类调用函数方法**，则这就是类方法，通过内置装饰器`@calssmethod`

```python
class cal:
    cal_name = '计算器'
    def __init__(self,x,y):
        self.x = x
        self.y = y

    @property           #在cal_add函数前加上@property，使得该函数可直接调用，封装起来
    def cal_add(self):
        return self.x + self.y

    @classmethod        #在cal_info函数前加上@classmethon，则该函数变为类方法，该函数只能访问到类的数据属性，不能获取实例的数据属性
    def cal_info(cls):  #python自动传入位置参数cls就是类本身
        print('这是一个%s'%cls.cal_name)   #cls.cal_name调用类自己的数据属性

cal.cal_info()   #>>> '这是一个计算器'
```

没有实例化类而直接调用类函数。

当然类方法，实例也可以调用，但是并没有什么用，违反了初衷：类方法就是专门供类使用

***@staticmethod 静态方法只是名义上归属类管理***，但是不能使用类变量和实例变量，是类的工具包
放在函数前（该函数不传入self或者cls），所以不能访问类属性和实例属性

```python
class cal:
    cal_name = '计算器'
    def __init__(self,x,y):
        self.x = x
        self.y = y

    @property           #在cal_add函数前加上@property，使得该函数可直接调用，封装起来
    def cal_add(self):
        return self.x + self.y

    @classmethod        #在cal_info函数前加上@classmethon，则该函数变为类方法，该函数只能访问到类的数据属性，不能获取实例的数据属性
    def cal_info(cls):  #python自动传入位置参数cls就是类本身
        print('这是一个%s'%cls.cal_name)   #cls.cal_name调用类自己的数据属性

    @staticmethod       #静态方法 类或实例均可调用
    def cal_test(a,b,c): #改静态方法函数里不传入self 或 cls
        print(a,b,c)
c1 = cal(10,11)
cal.cal_test(1,2,3)     #>>> 1 2 3
c1.cal_test(1,2,3)      #>>> 1 2 3
```

