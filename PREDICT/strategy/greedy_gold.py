import xlrd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 常量
bitcoin = 0
cash = 500
gold = 0
preperty = 500
days = []
preperties = [500,500,500,500,500]

def read_file(path):
    sheets = xlrd.open_workbook(path)
    sheet1 = sheets.sheet_by_index(0)
    cols = sheet1.col_values(1)[1:] # 真实值
    cols2 = sheet1.col_values(3)[1:] # 预测值
    cols2_half = sheet1.col_values(4)[1:] # 预测值
    cols3 = sheet1.col_values(5)[1:] # change rate
    cols4 = sheet1.col_values(6)[1:] # increase/de 1/0
    cols44 = sheet1.col_values(8)[6:-1] # increase/de 1/0
    cols5 = [0] # 连续跌n次 -n
    cols6 = [0] # 幅度
    length = len(cols[1:])
    for i in range(1,length):
        temp = int(cols4[i])
        state = 1 if temp == 1 else -1
        j = i
        while j >= 0:
            j = j - 1
            if (float(cols4[j]) * state > 0) or (state < 0 and cols4[j] == 0):
                if state > 0:
                    state =  state + 1
                else:
                    state =  state - 1
            else:
                break
        cols5.append(state)
        val = 0
        for q in range(abs(state)):
            val = val + cols3[i-q]
        cols6.append(val)

    return cols, cols2, cols3, cols4, cols5, cols6, cols2_half, cols44 # 前5个预测为0

def func(path):
    global cash
    global gold
    global days
    # col5 -> 连续跌n次 -n col6 累计变化率
    money = 0
    day = 0
    col1, col2, col3, col4, col5, col6, col_predict, col_predict_rate = read_file(path)
    col3.sort()
    length = len(col1)
    flag = 0
    continue_list = []
    for i in range(0,length-10):
        # print("第",i+5,"天")
        # print("cash",cash)
        # print("gold",gold)
        if i in continue_list:
            continue
        temp = cash + gold * col1[i+5]
        preperties.append(temp)
        rate = col_predict_rate[i]
        if rate > 0.03:
            print("col1",col1[i+5])
            print("col2",col1[i+6])
            # 交易
            gold = gold + cash*0.99/col1[i+5]
            cash = 0
            continue_list.append(i+1)
            cash = cash + gold*0.99*col1[i+6]
            gold = 0
    print("cash",cash)
    print("gold",gold)



def theory_max(path):
    col1, col2, col3, col4, col5, col6, col_predict, col_predict_rate = read_file(path)
    length = len(col1)
    low = min(col1)
    high = max(col1)
    print("low",low)
    print("high",high)


def main():
    path = "predict/gold.xls"
    func(path)
    print("cash",cash)
    print("gold",gold)
    print("days",days)
    temp = preperties[-1]
    for i in range(5):
        preperties.append(temp)
    print("properties",preperties)
    x = range(len(preperties))
    plt.plot(x,preperties)
    plt.xlabel("time/days")
    plt.ylabel("value/dollars")
    plt.title("value of gold investment")
    plt.show()
    theory_max(path)
if __name__ == "__main__":
    main()