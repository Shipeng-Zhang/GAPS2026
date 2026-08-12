import math


def acc(x1,x2,x3):
    avg = (x1+x2+x3)/3
    tmp = (x1-avg)**2+(x2-avg)**2+(x3-avg)**2
    var = math.sqrt(tmp/3)
    print(f"平均值为:{avg};方差为:{var}")

nums = [31.2,27.1,24.6]
acc(nums[0],nums[1],nums[2])