import xlrd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 常量
bitcoin = 0
cash = 500
gold = 0
property = 500
days = []
properties = [500,500,500,500,500]
rate3 = 0.02
rate2 = 1 - rate3

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
    global property
    global bitcoin
    # col5 -> 连续跌n次 -n col6 累计变化率
    day = 0
    col1, col2, col3, col4, col5, col6, col_predict, col_predict_rate = read_file(path)
    length = len(col1)
    flag = 0
    continue_list = []
    for i in range(0,length-10):
        property = col1[i+5] * bitcoin + cash
        print("第",i+5,"天",property)
        # print("第",i+5,"天")
        # print("bitcoin",bitcoin)
        # print("cash",cash)
        # print("property",property)
        temp = cash + bitcoin * col1[i+5]
        properties.append(temp)
        q = i + 5
        if q in continue_list:
            continue
        rate = 1
        state = 0
        for j in range(3):
            rate = rate * (1 + col_predict_rate[i+j])
        if col3[i+5] < 0: # 当前在下降
            if rate > (1+rate3):
                flag = 1
                day = i+3
                change = 0
                # stage of trade
                # define n constant times of increase/decrease
                index = i
                num = 0
                state = col5[i+5] # 连续第n次跌 -n
                if state > 4:
                    continue
                # print("index",i+6)
                # print("涨跌情况",state)
                # print("rate",rate)
                # 建立指数模型
                # 1 3 10 30 根据90% n=4
                base = property / (1+3+10+30)
                if state == 1:
                    change = base
                elif state == 2:
                    change = base * 3
                elif state == 3:
                    change = base * 10
                else:
                    change = base * 30
                if col3[i+5] >= 0:
                    change = change * (-1)
                if cash == 0:
                    continue
                if cash >= 0 :
                    if cash >= change:
                        cash = cash - change
                        bitcoin = bitcoin + change*(1-rate3)/col1[i+5]
                    else:
                        bitcoin = bitcoin + cash*(1-rate3)/col1[i+5]
                        cash = 0
                    print("第",i+5,"天交易-1") # 买入
                    print(col_predict_rate[i+4])
                    # print("当前money",money)
                continue
        if flag == 1 and i > day : # 没有卖出
            for m in range(3):
                if col3[i+m+5] < 0:
                    print("第",i+5+m,"天交易+1",) # 1519
                    print(col_predict_rate[i+4+m])
                    cash = cash + (1-rate3) * bitcoin * col1[i+4]
                    bitcoin = 0
                    flag = 0
                    break
        if rate > (1+rate*4):
            print("***")
            change = cash
            print("change",change)
            print("第",i+5,"天交易-2") # 1518
            print(col1[i+5])
            bitcoin = bitcoin + (1-rate3) * change/col1[i+5]
            cash = 0
            print("第",i+8,"天交易+2")
            print(col1[i+8])
            cash = cash + (1-rate3) * bitcoin * col1[i+8]
            bitcoin = 0
            for n in range(i+6,i+9):
                continue_list.append(n)
            continue
        # if rate<0.97:
        #     cash = cash + bitcoin * col1[i+5]
        #     bitcoin = 0
        #     print("第",i+5,"天交易+3")

def theory_max(path):
    col1, col2, col3, col4, col5, col6, col_predict, col_predict_rate = read_file(path)
    ans = maxProfit(len(col1),col1)
    print(ans)

def maxProfit(k, prices):
    k = min(k, len(prices) // 2)
    buy = [-float("inf")] * (k+1)  # 买入
    sell = [0] * (k+1)  # 卖出
    for p in prices:
        for i in range(1, k+1):
            buy[i] = max(buy[i], sell[i-1] - p*0.98)
            sell[i] = max(sell[i], buy[i] + p*0.98)
    temp = []
    for i in range(len(buy)):
        if sell[i]!=0:
            temp.append(abs(sell[i]/buy[i]))
    print(temp)

    return sell[-1]

def main():
    path = "predict/bit.xls"
    func(path)
    print("cash",cash)
    print("bitcoin",bitcoin)
    for i in range(5):
        properties.append(properties[-1])
    x = range(len(properties))
    plt.plot(x,properties)
    plt.xlabel("time/days")
    plt.ylabel("value/dollars")
    plt.title("return on bitcoin investment")
    plt.show()
    print("*******")
    print("理论最大值")
    theory_max(path)
    print("资产最大值",properties[-1])

if __name__ == "__main__":
    main()